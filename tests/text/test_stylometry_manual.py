from veridex.text import StylometricSignal

def test_stylometry():
    signal = StylometricSignal()
    
    # Test human-like text
    human_text = "This is a complex sentence with varied vocabulary, punctuation! and distinct structure."
    res_human = signal.run(human_text)
    
    assert 0.0 <= res_human.score <= 1.0, f"Score {res_human.score} out of bounds"
    assert res_human.metadata["token_count"] > 0
    
    # Test repetitive text (simulated AI failure mode)
    ai_text = "The cat is on the mat. The cat is on the mat. The cat is on the mat."
    res_ai = signal.run(ai_text)
    
    # Expect AI text to have higher score (lower TTR)
    # TTR low -> Score high
    assert res_ai.score > res_human.score, f"Expected AI score {res_ai.score} > Human score {res_human.score}"
    
    print("Stylometry tests passed!")

if __name__ == "__main__":
    test_stylometry()
