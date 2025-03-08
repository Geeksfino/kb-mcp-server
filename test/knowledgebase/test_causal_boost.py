"""Test custom causal boost configuration loading."""
import logging
import pytest
from pathlib import Path
from txtai_mcp_server.tools.causal_config import CausalBoostConfig
from txtai_mcp_server.tools.retrieve import detect_language

logger = logging.getLogger(__name__)

@pytest.fixture
def custom_config():
    """Load custom causal boost configuration for testing."""
    config_path = Path(__file__).parent / "causal_boost.yml"
    return CausalBoostConfig.load_from_file(config_path)

def test_config_loading(custom_config):
    """Test basic configuration loading."""
    assert custom_config is not None
    assert custom_config.default_language == "en"
    assert custom_config.supported_languages == {"en", "zh"}
    assert len(custom_config.boosts) == 5  # All boost multipliers present

def test_english_patterns(custom_config):
    """Test English pattern retrieval and content."""
    patterns = custom_config.get_patterns("en", "general")
    assert "causes" in patterns["keywords"]
    assert "not cause" in patterns["negation"]
    assert "why" in patterns["intent"]
    assert "the" in patterns["stopwords"]

    # Test technical domain
    tech_patterns = custom_config.get_patterns("en", "technical")
    assert "correlates with" in tech_patterns["keywords"]
    assert "no correlation" in tech_patterns["negation"]

def test_chinese_patterns(custom_config):
    """Test Chinese pattern retrieval and content."""
    patterns = custom_config.get_patterns("zh", "general")
    assert "导致" in patterns["keywords"]
    assert "不会导致" in patterns["negation"]
    assert "为什么" in patterns["intent"]
    assert "的" in patterns["stopwords"]

    # Test technical domain
    tech_patterns = custom_config.get_patterns("zh", "technical")
    assert "相关" in tech_patterns["keywords"]
    assert "没有相关" in tech_patterns["negation"]

def test_boost_values(custom_config):
    """Test custom boost multipliers."""
    assert custom_config.boosts["causal_intent"] == 1.4
    assert custom_config.boosts["general_query"] == 1.2
    assert custom_config.boosts["domain_specific"] == 1.3
    assert custom_config.boosts["multiple_term"] == 0.15
    assert custom_config.boosts["negation"] == 0.6

def test_language_detection():
    """Test language detection with sample queries."""
    # English queries
    en_result1 = detect_language("What causes climate change?")
    assert en_result1['lang'] == "en"
    assert en_result1['score'] > 0.0
    
    en_result2 = detect_language("How does temperature affect pressure?")
    assert en_result2['lang'] == "en"
    assert en_result2['score'] > 0.0

    # Chinese queries
    zh_result1 = detect_language("什么导致气候变化？")
    assert zh_result1['lang'] == "zh"
    assert zh_result1['score'] > 0.0
    
    zh_result2 = detect_language("温度如何影响压力？")
    assert zh_result2['lang'] == "zh"
    assert zh_result2['score'] > 0.0

def test_pattern_fallback(custom_config):
    """Test fallback to default language/domain when unsupported ones are requested."""
    # Test unsupported language fallback
    fr_patterns = custom_config.get_patterns("fr")
    en_patterns = custom_config.get_patterns("en")
    assert fr_patterns == en_patterns  # Should fall back to English

    # Test unsupported domain fallback
    custom_domain = custom_config.get_patterns("en", "custom")
    general_domain = custom_config.get_patterns("en", "general")
    assert custom_domain == general_domain  # Should fall back to general

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests with pytest
    pytest.main([__file__, "-v"])
