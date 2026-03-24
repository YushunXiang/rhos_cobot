import base64
import json
import logging
import math
import time
from typing import Any

import cv2
from openai import OpenAI

from examples.piper_real.planner_config import PlannerConfig


class PlannerResponseError(RuntimeError):
    """Raised when the planner returns an unusable response."""


class LLMNavigationPlanner:
    def __init__(self, ros_operator: Any, config: PlannerConfig) -> None:
        self.ros_operator = ros_operator
        self.config = config
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self._history: list[dict[str, Any]] = []
        self._usable_steps = 0
        self._consecutive_failures = 0

    def confirm_navigation_safety(self, task_prompt: str) -> bool:
        logging.warning(
            "Navigation safety confirmation required before moving TRACER. "
            "Review docs/tracer-2.0-user-manual-v2.0.3-2023.09.pdf, ensure the area is clear, "
            "keep the robot within sight, and verify the emergency stop is released."
        )
        logging.info("Planned task: %s", task_prompt)
        try:
            answer = input("Type 'yes' to allow navigation: ").strip().lower()
        except EOFError:
            answer = ""
        confirmed = answer == "yes"
        if confirmed:
            self._log_status("navigation_confirmation", {"confirmed": True})
        else:
            self._log_status("navigation_confirmation", {"confirmed": False})
            self.stop_base()
        return confirmed

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
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a navigation planner for an AgileX TRACER robot. "
                        "Return JSON only with no markdown. "
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
        content = response.choices[0].message.content
        if isinstance(content, list):
            raw_text = "".join(
                part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
                for part in content
            )
        else:
            raw_text = str(content)
        raw_text = raw_text.strip()
        raw_json = self._extract_json_text(raw_text)
        payload = json.loads(raw_json)
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

    def stop_base(self) -> None:
        self.ros_operator.robot_base_publish([0.0, 0.0])

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

        if abs(linear_x) > self.config.max_linear_vel:
            raise PlannerResponseError(
                f"planner linear velocity {linear_x} exceeds limit {self.config.max_linear_vel}"
            )
        if abs(angular_z) > self.config.max_angular_vel:
            raise PlannerResponseError(
                f"planner angular velocity {angular_z} exceeds limit {self.config.max_angular_vel}"
            )

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

    @staticmethod
    def _extract_json_text(raw_text: str) -> str:
        stripped = raw_text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            stripped = stripped.replace("json\n", "", 1).strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise PlannerResponseError("planner response did not contain JSON")
        return stripped[start : end + 1]
