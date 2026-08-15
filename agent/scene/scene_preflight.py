from pydantic import BaseModel, ConfigDict, Field, model_validator


class ScenePreflightIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    code: str
    message: str
    object_id: str | None = None


class ScenePreflightReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    passed: bool
    issues: tuple[ScenePreflightIssue, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_consistency(self):
        if self.passed != (not self.issues):
            raise ValueError("passed must be true if and only if issues is empty")
        return self
