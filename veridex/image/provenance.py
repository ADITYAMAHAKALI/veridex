from veridex.core.provenance import C2PASignal

class C2PAImageProvenance(C2PASignal):
    """
    C2PA Provenance checker for Image files.
    """
    @property
    def name(self) -> str:
        return "c2pa_image_provenance"
