import base64
import json
import logging
import math
import time
from typing import Any

import cv2

from examples.piper_real import base_safety
from examples.piper_real.llm_utils import extract_message_json_text
from examples.piper_real.planner_config import PlannerConfig


class PlannerResponseError(RuntimeError):
    """Raised when the planner returns an unusable response."""


class LLMNavigationPlanner:
    def __init__(self, ros_operator: Any, config: PlannerConfig) -> None:
        from openai import OpenAI

        self.ros_operator = ros_operator
        self.config = config
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self._history: list[dict[str, Any]] = []
        self._usable_steps = 0
        self._consecutive_failures = 0

    def capture_front_image(self) -> str:
        if not self.ros_operator.img_front_deque:
            raise PlannerResponseError("front camera frame unavailable")
        img_msg = self.ros_operator.img_front_deque[-1]
        frame = self.ros_operator.bridge.imgmsg_to_cv2(img_msg, "passthrough")
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise PlannerResponseError("failed to encode front camera frame")
        image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{image_b64}"

    def get_odometry(self) -> dict[str, float]:
        if not self.ros_operator.robot_base_deque:
            raise PlannerResponseError("odometry unavailable")
        odom = self.ros_operator.robot_base_deque[-1]
        position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
        )
        return {"x": float(position.x), "y": float(position.y), "yaw": float(yaw)}

    def query_llm(
        self,
        image_b64: str,
        task_prompt: str,
        odom: dict[str, float],
        history: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        history_text = json.dumps(history[-5:], ensure_ascii=False)
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a navigation planner for an AgileX TRACER robot. "
                        "Return JSON only with no markdown, no prose, and no thinking process. "
                        "Allowed outputs are: "
                        "{\"action\":\"move\",\"linear_x\":float,\"angular_z\":float,\"duration\":float,\"reasoning\":str} "
                        "or {\"action\":\"stop\",\"reason\":str}. "
                        f"Keep linear_x within +/-{self.config.max_linear_vel} m/s and angular_z within +/-{self.config.max_angular_vel} rad/s. "
                        "Use stop when the robot is in a usable operating position."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Task prompt: {task_prompt}\n"
                                f"Current odometry: {json.dumps(odom, ensure_ascii=False)}\n"
                                f"Recent history: {history_text}"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": image_b64}},
                    ],
                },
            ],
        )
        try:
            raw_text, raw_json = extract_message_json_text(response.choices[0].message)
        except ValueError as exc:
            raise PlannerResponseError(str(exc)) from exc
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise PlannerResponseError(f"planner returned invalid JSON: {raw_json[:400]}") from exc
        if not isinstance(payload, dict):
            raise PlannerResponseError("planner response must be a JSON object")
        return raw_text, payload

    def execute_command(self, cmd: dict[str, Any]) -> dict[str, float]:
        linear_x = float(cmd["linear_x"])
        angular_z = float(cmd["angular_z"])
        duration = float(cmd.get("duration", self.config.default_duration))
        self.ros_operator.robot_base_publish([linear_x, angular_z])
        time.sleep(max(duration, 0.0))
        self.stop_base()
        return {
            "linear_x": linear_x,
            "angular_z": angular_z,
            "duration": duration,
        }

    def _execute_command(self, cmd: dict[str, Any]) -> None:
        linear_x = float(cmd["linear_x"])
        angular_z = float(cmd["angular_z"])
        duration = float(cmd.get("duration", self.config.default_duration))
        control_hz = float(cmd.get("control_hz", 10.0))
        period = 1.0 / control_hz
        start = time.monotonic()
        next_tick = start
        while time.monotonic() - start < duration:
            self.ros_operator.robot_base_publish([linear_x, angular_z])
            next_tick += period
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # Keep a fixed-rate schedule without accumulating drift.
                next_tick = time.monotonic()

    def stop_base(self) -> None:
        base_safety.stop_base(self.ros_operator)

    def run(self, task_prompt: str) -> bool:
        self._history = []
        self._usable_steps = 0
        self._consecutive_failures = 0
        self.stop_base()

        cycle_index = 0
        while True:
            cycle_index += 1
            try:
                image_b64 = self.capture_front_image()
                odom = self.get_odometry()
            except PlannerResponseError as exc:
                self.stop_base()
                self._log_status(
                    "navigation_failed",
                    {"cycle": cycle_index, "reason": str(exc), "usable_steps": self._usable_steps},
                )
                return False

            raw_text = ""
            try:
                raw_text, payload = self.query_llm(image_b64, task_prompt, odom, self._history)
                normalized = self._normalize_command(payload)
                logging.info(
                    "Planner normalized decision [cycle %s]: %s",
                    cycle_index,
                    json.dumps(normalized, ensure_ascii=False),
                )
                self._consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001
                self._consecutive_failures += 1
                self.stop_base()
                self._append_event(
                    {
                        "cycle": cycle_index,
                        "type": "planner_failure",
                        "error": str(exc),
                        "raw_payload": raw_text,
                        "consecutive_failures": self._consecutive_failures,
                    }
                )
                logging.error("Planner failure on cycle %s: %s", cycle_index, exc)
                if self._consecutive_failures >= 4:
                    self._log_status(
                        "navigation_failed",
                        {
                            "cycle": cycle_index,
                            "reason": "planner failed 4 consecutive times",
                            "usable_steps": self._usable_steps,
                        },
                    )
                    return False
                continue

            self._usable_steps += 1
            logging.info("Planner raw response [cycle %s]: %s", cycle_index, raw_text)

            if normalized["action"] == "stop":
                self.stop_base()
                self._append_event(
                    {
                        "cycle": cycle_index,
                        "type": "stop",
                        "usable_steps": self._usable_steps,
                        "reason": normalized.get("reason", "planner requested stop"),
                        "raw_payload": raw_text,
                        "odom": odom,
                    }
                )
                self._log_status(
                    "navigation_succeeded",
                    {
                        "cycle": cycle_index,
                        "reason": normalized.get("reason", "planner requested stop"),
                        "usable_steps": self._usable_steps,
                    },
                )
                return True

            executed = self.execute_command(normalized)
            self._append_event(
                {
                    "cycle": cycle_index,
                    "type": "move",
                    "usable_steps": self._usable_steps,
                    "command": executed,
                    "reasoning": normalized.get("reasoning", ""),
                    "raw_payload": raw_text,
                    "odom": odom,
                }
            )

            if self._usable_steps >= self.config.max_nav_steps:
                self.stop_base()
                self._log_status(
                    "navigation_failed",
                    {
                        "cycle": cycle_index,
                        "reason": "usable planner step limit reached",
                        "usable_steps": self._usable_steps,
                    },
                )
                return False

    def run_routine(self, routine: list[tuple[float, float]]) -> bool:
        self._history = []
        self._usable_steps = 0
        self._consecutive_failures = 0
        self.stop_base()

        if not routine:
            self._append_event(
                {
                    "cycle": 0,
                    "type": "stop",
                    "usable_steps": self._usable_steps,
                    "reason": "routine is empty",
                    "raw_payload": "",
                }
            )
            self._log_status(
                "navigation_succeeded",
                {"cycle": 0, "reason": "routine is empty", "usable_steps": self._usable_steps},
            )
            return True

        # Use conservative open-loop speeds to reduce overshoot risk.
        linear_speed = 0.6 * self.config.max_linear_vel
        angular_speed = 0.6 * self.config.max_angular_vel

        cycle_index = 0
        for target_x_raw, target_yaw_raw in routine:
            cycle_index += 1
            try:
                odom = self.get_odometry()
            except PlannerResponseError as exc:
                self.stop_base()
                self._log_status(
                    "navigation_failed",
                    {
                        "cycle": cycle_index,
                        "reason": str(exc),
                        "usable_steps": self._usable_steps,
                    },
                )
                return False

            raw_text = ""
            try:
                target_x = float(target_x_raw)
                target_yaw = float(target_yaw_raw)

                linear_cmd = linear_speed if target_x > 0 else -linear_speed
                angular_cmd = angular_speed if target_yaw > 0 else -angular_speed
                duration_s = max(abs(target_x) / linear_speed, abs(target_yaw) / angular_speed)
                raw_text = json.dumps(
                    {
                        "action": "move",
                        "linear_x": linear_cmd,
                        "angular_z": angular_cmd,
                        "duration": duration_s,
                        "reasoning": "fixed routine step",
                    },
                    ensure_ascii=False,
                )
                normalized = self._normalize_command(json.loads(raw_text))
                logging.info(
                    "Planner normalized decision [cycle %s]: %s",
                    cycle_index,
                    json.dumps(normalized, ensure_ascii=False),
                )
                self._consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001
                self._consecutive_failures += 1
                self.stop_base()
                self._append_event(
                    {
                        "cycle": cycle_index,
                        "type": "planner_failure",
                        "error": str(exc),
                        "raw_payload": raw_text,
                        "consecutive_failures": self._consecutive_failures,
                    }
                )
                logging.error("Planner failure on cycle %s: %s", cycle_index, exc)
                if self._consecutive_failures >= 4:
                    self._log_status(
                        "navigation_failed",
                        {
                            "cycle": cycle_index,
                            "reason": "planner failed 4 consecutive times",
                            "usable_steps": self._usable_steps,
                        },
                    )
                    return False
                continue

            self._usable_steps += 1
            logging.info("Planner raw response [cycle %s]: %s", cycle_index, raw_text)

            executed = self.execute_command(normalized)
            self._append_event(
                {
                    "cycle": cycle_index,
                    "type": "move",
                    "usable_steps": self._usable_steps,
                    "command": executed,
                    "reasoning": normalized.get("reasoning", ""),
                    "raw_payload": raw_text,
                    "odom": odom,
                }
            )

            if self._usable_steps >= self.config.max_nav_steps:
                self.stop_base()
                self._log_status(
                    "navigation_failed",
                    {
                        "cycle": cycle_index,
                        "reason": "usable planner step limit reached",
                        "usable_steps": self._usable_steps,
                    },
                )
                return False

        self.stop_base()
        self._append_event(
            {
                "cycle": cycle_index,
                "type": "stop",
                "usable_steps": self._usable_steps,
                "reason": "fixed routine completed",
                "raw_payload": "",
            }
        )
        self._log_status(
            "navigation_succeeded",
            {
                "cycle": cycle_index,
                "reason": "fixed routine completed",
                "usable_steps": self._usable_steps,
            },
        )
        return True

    def _normalize_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = payload.get("action")
        if action == "stop":
            return {
                "action": "stop",
                "reason": str(payload.get("reason", "planner requested stop")),
            }
        if action != "move":
            raise PlannerResponseError(f"unsupported planner action: {action!r}")

        try:
            linear_x = float(payload["linear_x"])
            angular_z = float(payload["angular_z"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PlannerResponseError("move response missing numeric velocity fields") from exc

        try:
            linear_x, angular_z = base_safety.enforce_base_velocity_limits(
                linear_x,
                angular_z,
                max_linear_vel=self.config.max_linear_vel,
                max_angular_vel=self.config.max_angular_vel,
                source="planner command",
            )
        except ValueError as exc:
            raise PlannerResponseError(str(exc)) from exc

        duration = payload.get("duration", self.config.default_duration)
        try:
            duration_value = float(duration)
        except (TypeError, ValueError) as exc:
            raise PlannerResponseError("move duration must be numeric") from exc
        if duration_value <= 0:
            raise PlannerResponseError("move duration must be positive")

        return {
            "action": "move",
            "linear_x": linear_x,
            "angular_z": angular_z,
            "duration": duration_value,
            "reasoning": str(payload.get("reasoning", "")),
        }

    def _append_event(self, event: dict[str, Any]) -> None:
        self._history.append(event)
        logging.info("Navigation event: %s", json.dumps(event, ensure_ascii=False))

    def _log_status(self, status: str, details: dict[str, Any]) -> None:
        payload = {"status": status, **details}
        if status.endswith("failed"):
            logging.error("Navigation status: %s", json.dumps(payload, ensure_ascii=False))
        elif status.endswith("succeeded"):
            logging.info("Navigation status: %s", json.dumps(payload, ensure_ascii=False))
        else:
            logging.warning("Navigation status: %s", json.dumps(payload, ensure_ascii=False))
