import json
from typing import Any, Optional
from veridex.core.signal import BaseSignal, DetectionResult

class C2PASignal(BaseSignal):
    """
    Detects Content Credentials (C2PA) manifests in files.

    This signal checks if a file contains a C2PA manifest and parses it to determine
    if the content is cryptographically signed as AI-generated.

    Attributes:
        name (str): 'c2pa_provenance'
        dtype (str): 'file'

    Raises:
        ImportError: If `c2pa-python` is not installed.
    """

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "c2pa_provenance"

    @property
    def dtype(self) -> str:
        return "file"

    def check_dependencies(self) -> None:
        try:
            import c2pa
        except ImportError:
            raise ImportError(
                "The 'c2pa' library is required for C2PASignal. "
                "Install it with `pip install veridex[c2pa]`."
            )

    def run(self, input_data: Any) -> DetectionResult:
        """
        Analyzes the file for C2PA manifests.

        Args:
            input_data (str): Path to the file.

        Returns:
            DetectionResult:
                score=1.0 if AI assertion found.
                score=0.0 otherwise (Human assertion or No Manifest).
        """
        if not isinstance(input_data, str):
            return DetectionResult(score=0.0, confidence=0.0, metadata={}, error="Input must be a file path string.")

        try:
            self.check_dependencies()
            import c2pa
        except ImportError as e:
            return DetectionResult(
                score=0.0,
                confidence=0.0,
                metadata={},
                error=str(e)
            )

        try:
            # c2pa.Reader context manager
            with c2pa.Reader(input_data) as reader:
                manifest_data = reader.get_active_manifest()

                if not manifest_data:
                    return DetectionResult(
                        score=0.0,
                        confidence=0.0,
                        metadata={"status": "no_active_manifest"},
                        error=None
                    )

                assertions = manifest_data.get("assertions", [])

                # We look for AI indicators in assertions
                is_ai = False
                found_assertions = []

                for assertion in assertions:
                    label = assertion.get("label", "")
                    data = assertion.get("data", {})

                    found_assertions.append(label)

                    # Check 1: Digital Source Type (IPTC)
                    # Label: stds.iptc.digitalSourceType
                    # URI for AI: http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia
                    # URI for Composite AI: http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia
                    # URI for Algorithmic: http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicMedia
                    if label == "stds.iptc.digitalSourceType":
                        source_type = data.get("val", "")
                        if "trainedAlgorithmicMedia" in source_type or "algorithmicMedia" in source_type:
                            is_ai = True

                    # Check 2: C2PA Actions
                    # Label: c2pa.actions
                    # If an action indicates 'c2pa.action.created' with digitalSourceType parameter
                    if label.startswith("c2pa.actions"):
                        actions_list = data.get("actions", [])
                        for action in actions_list:
                            # Check digitalSourceType in action parameters
                            ds_type = action.get("digitalSourceType", "")
                            if "trainedAlgorithmicMedia" in ds_type or "algorithmicMedia" in ds_type:
                                is_ai = True

                            # Check for 'softwareAgent' that might be known AI (less reliable, but indicative)
                            # Avoiding hardcoded software names for now unless specified.

                return DetectionResult(
                    score=1.0 if is_ai else 0.0,
                    confidence=1.0, # Cryptographic signature is high confidence
                    metadata={
                        "manifest_found": True,
                        "is_ai_signed": is_ai,
                        "found_assertions": found_assertions
                    }
                )

        except Exception as e:
            # If c2pa.Reader fails (e.g. format not supported, file not found), treat as no manifest
            # But return error details in metadata
            return DetectionResult(
                score=0.0,
                confidence=0.0,
                metadata={"status": "read_error", "details": str(e)},
                error=None
            )
