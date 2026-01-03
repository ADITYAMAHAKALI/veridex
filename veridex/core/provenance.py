import os
from typing import Any, Optional
from veridex.core.signal import BaseSignal, DetectionResult

class C2PASignal(BaseSignal):
    """
    Detects Content Credentials (C2PA) manifests in files.

    This signal checks if a file contains a C2PA manifest (embedded or sidecar)
    and parses it to determine if the content is cryptographically signed as AI-generated.

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

    def _find_sidecar(self, file_path: str) -> Optional[str]:
        """
        Looks for a sidecar manifest file.
        Checks:
        1. file_path + ".c2pa" (e.g. data.txt.c2pa)
        2. file_path base + ".c2pa" (e.g. data.c2pa for data.txt)
        """
        # Option 1: Append .c2pa
        path1 = file_path + ".c2pa"
        if os.path.exists(path1):
            return path1

        # Option 2: Replace extension (or add to base)
        base, ext = os.path.splitext(file_path)
        if ext:
            path2 = base + ".c2pa"
            if os.path.exists(path2):
                return path2

        return None

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
            sidecar_path = self._find_sidecar(input_data)
            manifest_bytes = None

            if sidecar_path:
                try:
                    with open(sidecar_path, 'rb') as f:
                        manifest_bytes = f.read()
                except Exception as e:
                    return DetectionResult(
                        score=0.0,
                        confidence=0.0,
                        metadata={"status": "sidecar_read_error", "details": str(e)},
                        error=None
                    )

            # Initialize Reader with or without manifest_data
            # If manifest_bytes provided, it treats input_data as the asset to bind against.
            with c2pa.Reader(input_data, manifest_data=manifest_bytes) as reader:
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
                    if label == "stds.iptc.digitalSourceType":
                        source_type = data.get("val", "")
                        if "trainedAlgorithmicMedia" in source_type or "algorithmicMedia" in source_type:
                            is_ai = True

                    # Check 2: C2PA Actions
                    if label.startswith("c2pa.actions"):
                        actions_list = data.get("actions", [])
                        for action in actions_list:
                            ds_type = action.get("digitalSourceType", "")
                            if "trainedAlgorithmicMedia" in ds_type or "algorithmicMedia" in ds_type:
                                is_ai = True

                return DetectionResult(
                    score=1.0 if is_ai else 0.0,
                    confidence=1.0,
                    metadata={
                        "manifest_found": True,
                        "is_ai_signed": is_ai,
                        "found_assertions": found_assertions,
                        "sidecar_used": bool(sidecar_path)
                    }
                )

        except Exception as e:
            return DetectionResult(
                score=0.0,
                confidence=0.0,
                metadata={"status": "read_error", "details": str(e)},
                error=None
            )
