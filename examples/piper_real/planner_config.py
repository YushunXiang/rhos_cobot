import dataclasses

from examples.piper_real.base_safety import TRACER_MANUAL_MAX_ANGULAR_VEL_RAD_S
from examples.piper_real.base_safety import TRACER_MANUAL_MAX_LINEAR_VEL_MPS


@dataclasses.dataclass
class PlannerConfig:
    base_url: str = "http://192.168.3.123:8000/v1"
    model: str = "Qwen/Qwen3.5-4B"
    api_key: str = "EMPTY"
    max_nav_steps: int = 20
    max_linear_vel: float = 0.3
    max_angular_vel: float = 0.5
    default_duration: float = 1.5
    progress_complete_threshold: float = 0.85
    progress_stall_threshold: float = 0.02
    progress_stall_steps: int = 3
    progress_regression_threshold: float = 0.1
    progress_confirm_with_replanner: bool = False

    def validate_motion_limits(self) -> None:
        if self.max_nav_steps <= 0:
            raise ValueError("planner.max_nav_steps must be positive")
        if self.max_linear_vel <= 0:
            raise ValueError("planner.max_linear_vel must be positive")
        if self.max_linear_vel > TRACER_MANUAL_MAX_LINEAR_VEL_MPS:
            raise ValueError(
                f"planner.max_linear_vel must be <= {TRACER_MANUAL_MAX_LINEAR_VEL_MPS} m/s "
                "per the TRACER manual"
            )
        if self.max_angular_vel <= 0:
            raise ValueError("planner.max_angular_vel must be positive")
        if self.max_angular_vel > TRACER_MANUAL_MAX_ANGULAR_VEL_RAD_S:
            raise ValueError(
                f"planner.max_angular_vel must be <= {TRACER_MANUAL_MAX_ANGULAR_VEL_RAD_S} rad/s "
                "per the TRACER manual"
            )
        if self.default_duration <= 0:
            raise ValueError("planner.default_duration must be positive")
        if not 0.0 <= self.progress_complete_threshold <= 1.0:
            raise ValueError("planner.progress_complete_threshold must be in [0, 1]")
        if self.progress_stall_threshold < 0:
            raise ValueError("planner.progress_stall_threshold must be non-negative")
        if self.progress_stall_steps <= 0:
            raise ValueError("planner.progress_stall_steps must be positive")
        if self.progress_regression_threshold < 0:
            raise ValueError("planner.progress_regression_threshold must be non-negative")

    def validate_service_config(self) -> None:
        if not self.base_url.strip():
            raise ValueError("planner.base_url must be set when navigation is enabled")
        if not self.model.strip():
            raise ValueError("planner.model must be set when navigation is enabled")

    def validate(self) -> None:
        self.validate_motion_limits()
        self.validate_service_config()
