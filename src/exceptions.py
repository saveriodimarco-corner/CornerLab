class InsufficientDataError(ValueError):
    """Raised when a required runtime input is absent."""


class InvalidMatchDataError(ValueError):
    """Raised when required match data is malformed."""


class InvalidFeatureDataError(ValueError):
    """Raised when required model features are missing or invalid."""


class BankrollUnavailableError(ValueError):
    """Raised when settled history exists but current bankroll cannot be resolved safely."""