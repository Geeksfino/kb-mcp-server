"""
Configuration for causal boost feature with multilingual and domain support.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Set, Union, Optional
import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class CausalBoostConfig(BaseModel):
    """Configuration for causal boost feature with multilingual and domain support."""
    
    # Causal patterns by language and domain
    # Causal patterns by language and domain
    patterns: Dict[str, Dict[str, Dict[str, Union[Set[str], List[str]]]]] = {
        "en": {  # English
            "general": {  # General domain
                "keywords": {"causes", "leads to", "improves", "results in", "triggers", "impacts", "contributes to", "facilitates", "enables", "promotes", "enhances", "increases", "decreases", "reduces", "prevents", "blocks", "inhibits", "accelerates", "slows down", "influences"},
                "negation": ["not cause", "doesn't cause", "no evidence"],
                "intent": ["why", "how does", "what causes", "reason for"],
                "stopwords": {"what", "where", "when", "why", "how", "which", "who", "whose", "whom", 
                             "am", "is", "are", "was", "were", "be", "been", "being",
                             "have", "has", "had", "do", "does", "did",
                             "the", "a", "an", "this", "that", "these", "those",
                             "my", "your", "his", "her", "its", "our", "their",
                             "and", "or", "but", "if", "then", "else", "when",
                             "at", "by", "for", "with", "about", "against", "between", "into", "through",
                             "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
                             "again", "further", "then", "once", "here", "there", "all", "any", "both"}
            },
            "technical": {  # Technical domain
                "keywords": {"correlates with", "determines", "facilitates", "predicts", "indicates", "suggests", "implies", "demonstrates", "proves", "validates", "confirms", "establishes", "shows", "reveals", "signifies", "depends on", "relies on", "requires"},
                "negation": ["no correlation", "doesn't determine"],
                "intent": ["what determines", "how does it affect"],
                "stopwords": {"what", "where", "when", "why", "how", "which", "who", "whose", "whom", 
                             "am", "is", "are", "was", "were", "be", "been", "being",
                             "have", "has", "had", "do", "does", "did",
                             "the", "a", "an", "this", "that", "these", "those",
                             "my", "your", "his", "her", "its", "our", "their",
                             "and", "or", "but", "if", "then", "else", "when",
                             "at", "by", "for", "with", "about", "against", "between", "into", "through",
                             "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
                             "again", "further", "then", "once", "here", "there", "all", "any", "both"}
            }
        },
        "zh": {  # Chinese
            "general": {
                "keywords": {"导致", "引起", "造成", "改善", "提高", "触发", "影响", "促进", "助长", "推动", "加强", "增加", "减少", "降低", "防止", "阻止", "加速", "减缓", "影响", "促使"},
                "negation": ["不会导致", "没有证据", "不会引起"],
                "intent": ["为什么", "如何", "什么原因", "什么导致"],
                "stopwords": {"什么", "怎么", "如何", "为什么", "的", "了", "和", "或", "与", "及", "在", "是", "有", "这", "那", "一个", "一些", "一直", "不", "也", "很", "很多", "得", "都", "而", "且", "被", "把", "到", "向", "会", "能", "可以", "可能", "应该"}
            },
            "technical": {
                "keywords": {"相关", "决定", "促进", "影响", "预测", "指示", "说明", "暗示", "证明", "验证", "确认", "建立", "显示", "揭示", "表明", "取决于", "依赖于", "需要"},
                "negation": ["没有相关", "不相关", "不会决定"],
                "intent": ["什么决定", "如何影响"],
                "stopwords": {"什么", "怎么", "如何", "为什么", "的", "了", "和", "或", "与", "及", "在", "是", "有", "这", "那", "一个", "一些", "一直", "不", "也", "很", "很多", "得", "都", "而", "且", "被", "把", "到", "向", "会", "能", "可以", "可能", "应该"}
            }
        }
    }
    
    # Boost multipliers
    boosts: Dict[str, float] = {
        "causal_intent": 1.3,      # Boost for queries with explicit causal intent
        "general_query": 1.1,      # Base boost for causal matches
        "domain_specific": 1.2,    # Additional boost for domain-specific matches
        "multiple_term": 0.1,      # Per-term boost for multiple matches (max: 1.2)
        "negation": 0.7           # Penalty for negated causality
    }
    
    # Default settings
    default_language: str = "en"
    default_domain: str = "general"
    supported_languages: Set[str] = {"en", "zh"}
    
    @classmethod
    def load_from_file(cls, config_path: Union[str, Path]) -> "CausalBoostConfig":
        """Load configuration from a YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}. Using default configuration.")
            return cls()
            
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            
            # Convert keyword lists to sets
            if "patterns" in config_data:
                for lang in config_data["patterns"]:
                    for domain in config_data["patterns"][lang]:
                        if "keywords" in config_data["patterns"][lang][domain]:
                            config_data["patterns"][lang][domain]["keywords"] = set(
                                config_data["patterns"][lang][domain]["keywords"]
                            )
            
            if "supported_languages" in config_data:
                config_data["supported_languages"] = set(config_data["supported_languages"])
                    
            return cls(**config_data)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return cls()
    
    def get_patterns(self, language: Optional[str] = None, domain: Optional[str] = None) -> Dict[str, Union[Set[str], List[str]]]:
        """Get patterns for the specified language and domain."""
        lang = language or self.default_language
        dom = domain or self.default_domain
        
        # Get patterns with fallbacks
        lang_patterns = self.patterns.get(lang, self.patterns.get(self.default_language, {}))
        return lang_patterns.get(dom, lang_patterns.get(self.default_domain, {"keywords": set(), "negation": [], "intent": []}))

# Default configuration
DEFAULT_CAUSAL_CONFIG = CausalBoostConfig()

# Try to load configuration from environment variable
CAUSAL_CONFIG_PATH = os.environ.get("TXTAI_CAUSAL_CONFIG")
if CAUSAL_CONFIG_PATH:
    try:
        DEFAULT_CAUSAL_CONFIG = CausalBoostConfig.load_from_file(CAUSAL_CONFIG_PATH)
        logger.info(f"Loaded causal boost configuration from {CAUSAL_CONFIG_PATH}")
    except Exception as e:
        logger.warning(f"Failed to load causal boost configuration from {CAUSAL_CONFIG_PATH}: {e}")
