import os
import json
import mimetypes
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

        # Stream Adapter Class required for c2pa-python < 0.9.0 (e.g. 0.8.0)
        class StreamAdapter(c2pa.Stream):
            def __init__(self, file):
                self.file = file

            def read_stream(self, length):
                return self.file.read(length)

            def seek_stream(self, pos, mode):
                if mode == c2pa.SeekMode.START:
                    whence = os.SEEK_SET
                elif mode == c2pa.SeekMode.CURRENT:
                    whence = os.SEEK_CUR
                elif mode == c2pa.SeekMode.END:
                    whence = os.SEEK_END
                else:
                    raise ValueError(f"Unknown seek mode: {mode}")
                return self.file.seek(pos, whence)

            def write_stream(self, data):
                return self.file.write(data)

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

            # Determine format
            # c2pa expects MIME type or extension.
            mime_type, _ = mimetypes.guess_type(input_data)
            format_str = mime_type if mime_type else os.path.splitext(input_data)[1].lstrip('.')
            if not format_str:
                 # Fallback
                format_str = "application/octet-stream" 

            reader = c2pa.Reader()
            
            with open(input_data, 'rb') as f:
                adapter = StreamAdapter(f)
                
                json_result = None
                if manifest_bytes:
                    json_result = reader.from_manifest_data_and_stream(manifest_bytes, format_str, adapter)
                else:
                    json_result = reader.from_stream(format_str, adapter)

                if not json_result:
                     return DetectionResult(
                        score=0.0,
                        confidence=0.0,
                        metadata={"status": "no_active_manifest"},
                        error=None
                    )
                
                parsed_data = json.loads(json_result)
                active_manifest = parsed_data.get("active_manifest")

                if not active_manifest:
                    return DetectionResult(
                        score=0.0,
                        confidence=0.0,
                        metadata={"status": "no_active_manifest"},
                        error=None
                    )
                
                # 'active_manifest' is an ID string, we need to find the manifest object
                manifests = parsed_data.get("manifests", {})
                current_manifest = manifests.get(active_manifest)
                
                if not current_manifest:
                     return DetectionResult(
                        score=0.0,
                        confidence=0.0,
                        metadata={"status": "active_manifest_not_found_in_store"},
                        error=None
                    )

                assertions = current_manifest.get("assertions", [])

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
            # c2pa might raise errors if no manifest found in some versions/cases, 
            # or if file format issue.
            # Assuming if verification fails or no manifest, we land here or return above.
            return DetectionResult(
                score=0.0,
                confidence=0.0,
                metadata={"status": "read_error", "details": str(e)},
                error=None
            )
