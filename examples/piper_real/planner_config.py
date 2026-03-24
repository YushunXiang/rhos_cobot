import dataclasses


@dataclasses.dataclass
class PlannerConfig:
    base_url: str = "http://localhost:8000/v1"
    model: str = "qwen2.5-vl-72b"
    api_key: str = "EMPTY"
    max_nav_steps: int = 20
    max_linear_vel: float = 0.3
    max_angular_vel: float = 0.5
    default_duration: float = 1.5
    enable_navigation: bool = True

    def validate(self) -> None:
        if self.enable_navigation:
            if not self.base_url.strip():
                raise ValueError("planner.base_url must be set when navigation is enabled")
            if not self.model.strip():
                raise ValueError("planner.model must be set when navigation is enabled")
        if self.max_nav_steps <= 0:
            raise ValueError("planner.max_nav_steps must be positive")
        if self.max_linear_vel <= 0:
            raise ValueError("planner.max_linear_vel must be positive")
        if self.max_angular_vel <= 0:
            raise ValueError("planner.max_angular_vel must be positive")
        if self.default_duration <= 0:
            raise ValueError("planner.default_duration must be positive")
