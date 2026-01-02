import numpy as np
from PIL import Image
from veridex.image import ELASignal

def test_ela():
    signal = ELASignal()
    
    # Create a dummy image (noise)
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype='uint8'))
    
    res = signal.run(img)
    
    assert 0.0 <= res.score <= 1.0
    assert "ela_mean_diff" in res.metadata
    
    print("ELA tests passed!")

if __name__ == "__main__":
    test_ela()
