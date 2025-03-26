class VerificationResult:
    id: str
    verified: bool
    error: str

    def __init__(self, id: str, verified: bool, error: str = ""):
        self.id = id
        self.verified = verified
        self.error = error
