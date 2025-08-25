"""
Internationalization (i18n) Support for AI Hardware Co-Design Platform.

This module provides comprehensive internationalization capabilities including
multi-language support, localization, cultural adaptations, and global deployment readiness.
"""

import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages for internationalization."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"


class MessageCategory(Enum):
    """Categories of internationalized messages."""
    GENERAL = "general"
    TECHNICAL = "technical"
    ERROR_MESSAGES = "error_messages"
    SUCCESS_MESSAGES = "success_messages"
    WARNINGS = "warnings"
    OPTIMIZATION_TERMS = "optimization_terms"
    HARDWARE_TERMS = "hardware_terms"
    RESEARCH_TERMS = "research_terms"
    USER_INTERFACE = "user_interface"
    DOCUMENTATION = "documentation"


@dataclass
class LocalizationContext:
    """Context for localization including cultural preferences."""
    
    language: SupportedLanguage
    region: Optional[str] = None
    currency: str = "USD"
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    number_format: str = "en_US"
    measurement_system: str = "metric"  # metric or imperial
    text_direction: str = "ltr"  # ltr (left-to-right) or rtl (right-to-left)
    cultural_adaptations: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize language-specific defaults."""
        if self.language in [SupportedLanguage.ARABIC]:
            self.text_direction = "rtl"
        
        # Set cultural defaults
        if self.language == SupportedLanguage.JAPANESE:
            self.date_format = "%Y年%m月%d日"
            self.cultural_adaptations.update({
                "honorifics": True,
                "formal_language": True
            })
        elif self.language == SupportedLanguage.GERMAN:
            self.date_format = "%d.%m.%Y"
            self.number_format = "de_DE"
        elif self.language == SupportedLanguage.FRENCH:
            self.date_format = "%d/%m/%Y"
            self.number_format = "fr_FR"


@dataclass
class TranslationEntry:
    """Entry for translation with metadata."""
    
    key: str
    original_text: str
    translated_text: str
    language: SupportedLanguage
    category: MessageCategory
    context: Optional[str] = None
    technical_notes: List[str] = field(default_factory=list)
    reviewed: bool = False
    reviewer: Optional[str] = None
    last_updated: Optional[str] = None
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "original_text": self.original_text,
            "translated_text": self.translated_text,
            "language": self.language.value,
            "category": self.category.value,
            "context": self.context,
            "technical_notes": self.technical_notes,
            "reviewed": self.reviewed,
            "reviewer": self.reviewer,
            "last_updated": self.last_updated,
            "usage_count": self.usage_count
        }


class InternationalizationManager:
    """Manager for internationalization and localization."""
    
    def __init__(self):
        """Initialize internationalization manager."""
        self.translations: Dict[SupportedLanguage, Dict[str, TranslationEntry]] = defaultdict(dict)
        self.current_language = SupportedLanguage.ENGLISH
        self.localization_context = LocalizationContext(self.current_language)
        self.fallback_language = SupportedLanguage.ENGLISH
        
        # Load default translations
        self._load_default_translations()
        
        logger.info(f"Internationalization manager initialized with {len(self.translations)} languages")
    
    def _load_default_translations(self) -> None:
        """Load default translations for all supported languages."""
        
        # Define core technical translations
        core_translations = {
            # General terms
            "optimization": {
                SupportedLanguage.ENGLISH: "Optimization",
                SupportedLanguage.SPANISH: "Optimización",
                SupportedLanguage.FRENCH: "Optimisation",
                SupportedLanguage.GERMAN: "Optimierung",
                SupportedLanguage.JAPANESE: "最適化",
                SupportedLanguage.CHINESE_SIMPLIFIED: "优化",
                SupportedLanguage.CHINESE_TRADITIONAL: "優化",
                SupportedLanguage.KOREAN: "최적화",
                SupportedLanguage.PORTUGUESE: "Otimização",
                SupportedLanguage.ITALIAN: "Ottimizzazione",
                SupportedLanguage.RUSSIAN: "Оптимизация",
                SupportedLanguage.ARABIC: "تحسين",
                SupportedLanguage.HINDI: "अनुकूलन"
            },
            
            # Hardware terms
            "accelerator": {
                SupportedLanguage.ENGLISH: "Accelerator",
                SupportedLanguage.SPANISH: "Acelerador",
                SupportedLanguage.FRENCH: "Accélérateur", 
                SupportedLanguage.GERMAN: "Beschleuniger",
                SupportedLanguage.JAPANESE: "アクセラレータ",
                SupportedLanguage.CHINESE_SIMPLIFIED: "加速器",
                SupportedLanguage.CHINESE_TRADITIONAL: "加速器",
                SupportedLanguage.KOREAN: "가속기",
                SupportedLanguage.PORTUGUESE: "Acelerador",
                SupportedLanguage.ITALIAN: "Acceleratore",
                SupportedLanguage.RUSSIAN: "Ускоритель",
                SupportedLanguage.ARABIC: "مسرع",
                SupportedLanguage.HINDI: "त्वरक"
            },
            
            "neural_network": {
                SupportedLanguage.ENGLISH: "Neural Network",
                SupportedLanguage.SPANISH: "Red Neuronal",
                SupportedLanguage.FRENCH: "Réseau de Neurones",
                SupportedLanguage.GERMAN: "Neuronales Netzwerk",
                SupportedLanguage.JAPANESE: "ニューラルネットワーク",
                SupportedLanguage.CHINESE_SIMPLIFIED: "神经网络",
                SupportedLanguage.CHINESE_TRADITIONAL: "神經網絡",
                SupportedLanguage.KOREAN: "신경망",
                SupportedLanguage.PORTUGUESE: "Rede Neural",
                SupportedLanguage.ITALIAN: "Rete Neurale",
                SupportedLanguage.RUSSIAN: "Нейронная сеть",
                SupportedLanguage.ARABIC: "الشبكة العصبية",
                SupportedLanguage.HINDI: "न्यूरल नेटवर्क"
            },
            
            "performance": {
                SupportedLanguage.ENGLISH: "Performance",
                SupportedLanguage.SPANISH: "Rendimiento",
                SupportedLanguage.FRENCH: "Performance",
                SupportedLanguage.GERMAN: "Leistung",
                SupportedLanguage.JAPANESE: "性能",
                SupportedLanguage.CHINESE_SIMPLIFIED: "性能",
                SupportedLanguage.CHINESE_TRADITIONAL: "性能",
                SupportedLanguage.KOREAN: "성능",
                SupportedLanguage.PORTUGUESE: "Desempenho",
                SupportedLanguage.ITALIAN: "Prestazione",
                SupportedLanguage.RUSSIAN: "Производительность",
                SupportedLanguage.ARABIC: "الأداء",
                SupportedLanguage.HINDI: "प्रदर्शन"
            },
            
            # Status messages
            "optimization_started": {
                SupportedLanguage.ENGLISH: "Optimization started",
                SupportedLanguage.SPANISH: "Optimización iniciada",
                SupportedLanguage.FRENCH: "Optimisation démarrée",
                SupportedLanguage.GERMAN: "Optimierung gestartet",
                SupportedLanguage.JAPANESE: "最適化を開始しました",
                SupportedLanguage.CHINESE_SIMPLIFIED: "优化已开始",
                SupportedLanguage.CHINESE_TRADITIONAL: "優化已開始",
                SupportedLanguage.KOREAN: "최적화가 시작되었습니다",
                SupportedLanguage.PORTUGUESE: "Otimização iniciada",
                SupportedLanguage.ITALIAN: "Ottimizzazione avviata",
                SupportedLanguage.RUSSIAN: "Оптимизация запущена",
                SupportedLanguage.ARABIC: "بدأ التحسين",
                SupportedLanguage.HINDI: "अनुकूलन शुरू किया गया"
            },
            
            "optimization_completed": {
                SupportedLanguage.ENGLISH: "Optimization completed successfully",
                SupportedLanguage.SPANISH: "Optimización completada exitosamente",
                SupportedLanguage.FRENCH: "Optimisation terminée avec succès",
                SupportedLanguage.GERMAN: "Optimierung erfolgreich abgeschlossen",
                SupportedLanguage.JAPANESE: "最適化が正常に完了しました",
                SupportedLanguage.CHINESE_SIMPLIFIED: "优化成功完成",
                SupportedLanguage.CHINESE_TRADITIONAL: "優化成功完成",
                SupportedLanguage.KOREAN: "최적화가 성공적으로 완료되었습니다",
                SupportedLanguage.PORTUGUESE: "Otimização concluída com sucesso",
                SupportedLanguage.ITALIAN: "Ottimizzazione completata con successo",
                SupportedLanguage.RUSSIAN: "Оптимизация успешно завершена",
                SupportedLanguage.ARABIC: "تم التحسين بنجاح",
                SupportedLanguage.HINDI: "अनुकूलन सफलतापूर्वक पूरा हुआ"
            },
            
            # Error messages
            "optimization_failed": {
                SupportedLanguage.ENGLISH: "Optimization failed",
                SupportedLanguage.SPANISH: "La optimización falló",
                SupportedLanguage.FRENCH: "L'optimisation a échoué",
                SupportedLanguage.GERMAN: "Optimierung fehlgeschlagen",
                SupportedLanguage.JAPANESE: "最適化が失敗しました",
                SupportedLanguage.CHINESE_SIMPLIFIED: "优化失败",
                SupportedLanguage.CHINESE_TRADITIONAL: "優化失敗",
                SupportedLanguage.KOREAN: "최적화가 실패했습니다",
                SupportedLanguage.PORTUGUESE: "Otimização falhada",
                SupportedLanguage.ITALIAN: "Ottimizzazione fallita",
                SupportedLanguage.RUSSIAN: "Оптимизация не удалась",
                SupportedLanguage.ARABIC: "فشل التحسين",
                SupportedLanguage.HINDI: "अनुकूलन असफल"
            },
            
            "invalid_parameters": {
                SupportedLanguage.ENGLISH: "Invalid parameters provided",
                SupportedLanguage.SPANISH: "Parámetros inválidos proporcionados",
                SupportedLanguage.FRENCH: "Paramètres invalides fournis",
                SupportedLanguage.GERMAN: "Ungültige Parameter bereitgestellt",
                SupportedLanguage.JAPANESE: "無効なパラメータが提供されました",
                SupportedLanguage.CHINESE_SIMPLIFIED: "提供的参数无效",
                SupportedLanguage.CHINESE_TRADITIONAL: "提供的參數無效",
                SupportedLanguage.KOREAN: "잘못된 매개변수가 제공됨",
                SupportedLanguage.PORTUGUESE: "Parâmetros inválidos fornecidos",
                SupportedLanguage.ITALIAN: "Parametri non validi forniti",
                SupportedLanguage.RUSSIAN: "Предоставлены недействительные параметры",
                SupportedLanguage.ARABIC: "تم توفير معاملات غير صالحة",
                SupportedLanguage.HINDI: "अमान्य पैरामीटर प्रदान किए गए"
            }
        }
        
        # Load translations into the system
        for key, translations in core_translations.items():
            for language, text in translations.items():
                category = self._determine_category(key)
                
                translation_entry = TranslationEntry(
                    key=key,
                    original_text=translations[SupportedLanguage.ENGLISH],
                    translated_text=text,
                    language=language,
                    category=category,
                    reviewed=True,  # Core translations are pre-reviewed
                    reviewer="system"
                )
                
                self.translations[language][key] = translation_entry
        
        logger.info(f"Loaded {len(core_translations)} core translation keys")
    
    def _determine_category(self, key: str) -> MessageCategory:
        """Determine category based on translation key."""
        if any(term in key.lower() for term in ["error", "fail", "invalid"]):
            return MessageCategory.ERROR_MESSAGES
        elif any(term in key.lower() for term in ["success", "completed", "done"]):
            return MessageCategory.SUCCESS_MESSAGES
        elif any(term in key.lower() for term in ["warn", "caution", "alert"]):
            return MessageCategory.WARNINGS
        elif any(term in key.lower() for term in ["optimize", "algorithm", "search"]):
            return MessageCategory.OPTIMIZATION_TERMS
        elif any(term in key.lower() for term in ["accelerator", "hardware", "compute"]):
            return MessageCategory.HARDWARE_TERMS
        elif any(term in key.lower() for term in ["research", "study", "analysis"]):
            return MessageCategory.RESEARCH_TERMS
        elif any(term in key.lower() for term in ["ui", "interface", "button", "menu"]):
            return MessageCategory.USER_INTERFACE
        elif any(term in key.lower() for term in ["doc", "help", "guide"]):
            return MessageCategory.DOCUMENTATION
        elif any(term in key.lower() for term in ["neural", "network", "model"]):
            return MessageCategory.TECHNICAL
        else:
            return MessageCategory.GENERAL
    
    def set_language(self, language: SupportedLanguage, region: Optional[str] = None) -> None:
        """Set current language and update localization context."""
        self.current_language = language
        self.localization_context = LocalizationContext(language, region)
        
        logger.info(f"Language set to {language.value}")
    
    def translate(
        self, 
        key: str, 
        language: Optional[SupportedLanguage] = None,
        **kwargs
    ) -> str:
        """Translate a message key to the specified or current language."""
        
        target_language = language or self.current_language
        
        # Get translation
        if target_language in self.translations and key in self.translations[target_language]:
            translation_entry = self.translations[target_language][key]
            translation_entry.usage_count += 1
            translated_text = translation_entry.translated_text
        else:
            # Fallback to English or original key
            if self.fallback_language in self.translations and key in self.translations[self.fallback_language]:
                translated_text = self.translations[self.fallback_language][key].translated_text
            else:
                translated_text = key  # Last resort: return the key itself
        
        # Apply string formatting if kwargs provided
        if kwargs:
            try:
                translated_text = translated_text.format(**kwargs)
            except (KeyError, ValueError) as e:
                logger.warning(f"Translation formatting failed for key '{key}': {e}")
        
        return translated_text
    
    def add_translation(
        self, 
        key: str, 
        text: str, 
        language: SupportedLanguage,
        category: MessageCategory = MessageCategory.GENERAL,
        context: Optional[str] = None,
        technical_notes: Optional[List[str]] = None
    ) -> None:
        """Add a new translation entry."""
        
        # Get original text (English version)
        original_text = text
        if language != SupportedLanguage.ENGLISH:
            if (SupportedLanguage.ENGLISH in self.translations and 
                key in self.translations[SupportedLanguage.ENGLISH]):
                original_text = self.translations[SupportedLanguage.ENGLISH][key].translated_text
        
        translation_entry = TranslationEntry(
            key=key,
            original_text=original_text,
            translated_text=text,
            language=language,
            category=category,
            context=context,
            technical_notes=technical_notes or [],
            reviewed=False,  # New translations need review
            last_updated=self._get_current_timestamp()
        )
        
        self.translations[language][key] = translation_entry
        logger.info(f"Added translation for key '{key}' in language '{language.value}'")
    
    def get_supported_languages(self) -> List[SupportedLanguage]:
        """Get list of supported languages."""
        return list(SupportedLanguage)
    
    def get_translation_coverage(self, language: SupportedLanguage) -> Dict[str, Any]:
        """Get translation coverage statistics for a language."""
        
        if language not in self.translations:
            return {"coverage": 0.0, "total_keys": 0, "translated_keys": 0, "reviewed_translations": 0}
        
        english_keys = set(self.translations.get(SupportedLanguage.ENGLISH, {}).keys())
        language_keys = set(self.translations[language].keys())
        
        translated_keys = len(language_keys)
        total_keys = len(english_keys) if english_keys else translated_keys
        
        # Count reviewed translations
        reviewed_translations = sum(
            1 for entry in self.translations[language].values() if entry.reviewed
        )
        
        coverage = (translated_keys / total_keys * 100) if total_keys > 0 else 0.0
        
        return {
            "coverage": coverage,
            "total_keys": total_keys,
            "translated_keys": translated_keys,
            "reviewed_translations": reviewed_translations,
            "review_coverage": (reviewed_translations / translated_keys * 100) if translated_keys > 0 else 0.0
        }
    
    def format_number(self, number: Union[int, float], language: Optional[SupportedLanguage] = None) -> str:
        """Format number according to language/cultural conventions."""
        
        target_language = language or self.current_language
        
        # Basic formatting based on language
        if target_language in [SupportedLanguage.GERMAN, SupportedLanguage.FRENCH]:
            # Use comma as decimal separator, period for thousands
            if isinstance(number, float):
                return f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            else:
                return f"{number:,}".replace(",", ".")
        elif target_language in [SupportedLanguage.CHINESE_SIMPLIFIED, SupportedLanguage.CHINESE_TRADITIONAL]:
            # Chinese number formatting
            if number >= 10000:
                return f"{number/10000:.1f}万"
            else:
                return f"{number:,.0f}"
        else:
            # Default English formatting
            if isinstance(number, float):
                return f"{number:,.2f}"
            else:
                return f"{number:,}"
    
    def format_percentage(self, value: float, language: Optional[SupportedLanguage] = None) -> str:
        """Format percentage according to language conventions."""
        
        target_language = language or self.current_language
        formatted_number = self.format_number(value * 100, target_language)
        
        # Add percentage symbol
        if target_language == SupportedLanguage.FRENCH:
            return f"{formatted_number} %"  # Space before %
        else:
            return f"{formatted_number}%"
    
    def get_text_direction(self, language: Optional[SupportedLanguage] = None) -> str:
        """Get text direction (ltr or rtl) for language."""
        
        target_language = language or self.current_language
        
        if target_language == SupportedLanguage.ARABIC:
            return "rtl"
        else:
            return "ltr"
    
    def export_translations(self, language: SupportedLanguage, format: str = "json") -> str:
        """Export translations for a language."""
        
        if language not in self.translations:
            return "{}" if format == "json" else ""
        
        translations_dict = {}
        for key, entry in self.translations[language].items():
            translations_dict[key] = entry.to_dict()
        
        if format == "json":
            return json.dumps(translations_dict, indent=2, ensure_ascii=False)
        elif format == "csv":
            # Simple CSV format
            lines = ["key,original_text,translated_text,category,reviewed"]
            for entry in translations_dict.values():
                lines.append(f'"{entry["key"]}","{entry["original_text"]}","{entry["translated_text"]}","{entry["category"]}",{entry["reviewed"]}')
            return "\\n".join(lines)
        else:
            return str(translations_dict)
    
    def import_translations(self, language: SupportedLanguage, data: str, format: str = "json") -> int:
        """Import translations from external source."""
        
        imported_count = 0
        
        try:
            if format == "json":
                translations_data = json.loads(data)
                
                for key, entry_data in translations_data.items():
                    if isinstance(entry_data, dict):
                        # Full translation entry
                        translation_entry = TranslationEntry(
                            key=key,
                            original_text=entry_data.get("original_text", ""),
                            translated_text=entry_data.get("translated_text", ""),
                            language=language,
                            category=MessageCategory(entry_data.get("category", "general")),
                            context=entry_data.get("context"),
                            technical_notes=entry_data.get("technical_notes", []),
                            reviewed=entry_data.get("reviewed", False),
                            reviewer=entry_data.get("reviewer"),
                            last_updated=self._get_current_timestamp()
                        )
                    else:
                        # Simple key-value format
                        translation_entry = TranslationEntry(
                            key=key,
                            original_text=str(entry_data),
                            translated_text=str(entry_data),
                            language=language,
                            category=self._determine_category(key),
                            last_updated=self._get_current_timestamp()
                        )
                    
                    self.translations[language][key] = translation_entry
                    imported_count += 1
            
            logger.info(f"Imported {imported_count} translations for language {language.value}")
            
        except Exception as e:
            logger.error(f"Failed to import translations: {e}")
        
        return imported_count
    
    def get_translation_stats(self) -> Dict[str, Any]:
        """Get comprehensive translation statistics."""
        
        stats = {
            "supported_languages": len(self.get_supported_languages()),
            "languages_with_translations": len(self.translations),
            "total_translation_keys": 0,
            "language_coverage": {},
            "category_distribution": defaultdict(int),
            "review_status": {
                "reviewed": 0,
                "pending_review": 0
            },
            "most_used_translations": [],
            "current_language": self.current_language.value
        }
        
        all_keys = set()
        usage_counts = []
        
        for language, translations in self.translations.items():
            all_keys.update(translations.keys())
            stats["language_coverage"][language.value] = self.get_translation_coverage(language)
            
            for entry in translations.values():
                stats["category_distribution"][entry.category.value] += 1
                
                if entry.reviewed:
                    stats["review_status"]["reviewed"] += 1
                else:
                    stats["review_status"]["pending_review"] += 1
                
                if entry.usage_count > 0:
                    usage_counts.append((entry.key, entry.usage_count, language.value))
        
        stats["total_translation_keys"] = len(all_keys)
        
        # Get most used translations
        usage_counts.sort(key=lambda x: x[1], reverse=True)
        stats["most_used_translations"] = usage_counts[:10]
        
        return stats
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp as string."""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def validate_translation_completeness(self) -> Dict[str, List[str]]:
        """Validate translation completeness across all languages."""
        
        # Get all keys from English (reference language)
        english_keys = set(self.translations.get(SupportedLanguage.ENGLISH, {}).keys())
        
        missing_translations = {}
        
        for language in SupportedLanguage:
            if language == SupportedLanguage.ENGLISH:
                continue
                
            language_keys = set(self.translations.get(language, {}).keys())
            missing_keys = english_keys - language_keys
            
            if missing_keys:
                missing_translations[language.value] = list(missing_keys)
        
        return missing_translations
    
    def get_localized_message(self, message_key: str, **kwargs) -> str:
        """Get localized message with current language and formatting."""
        return self.translate(message_key, **kwargs)


# Global internationalization manager
_i18n_manager: Optional[InternationalizationManager] = None


def get_i18n_manager() -> InternationalizationManager:
    """Get internationalization manager instance."""
    global _i18n_manager
    
    if _i18n_manager is None:
        _i18n_manager = InternationalizationManager()
    
    return _i18n_manager


def translate(key: str, language: Optional[SupportedLanguage] = None, **kwargs) -> str:
    """Convenience function for translation."""
    manager = get_i18n_manager()
    return manager.translate(key, language, **kwargs)


def set_language(language: SupportedLanguage, region: Optional[str] = None) -> None:
    """Convenience function to set language."""
    manager = get_i18n_manager()
    manager.set_language(language, region)


def get_supported_languages() -> List[str]:
    """Get list of supported language codes."""
    return [lang.value for lang in SupportedLanguage]


# Translation decorator for automatic internationalization
def translatable(message_key: str, category: MessageCategory = MessageCategory.GENERAL):
    """Decorator to mark functions/methods for automatic translation."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # If result is a string, try to translate it
            if isinstance(result, str) and message_key:
                manager = get_i18n_manager()
                
                # Add the translation if it doesn't exist
                if message_key not in manager.translations.get(SupportedLanguage.ENGLISH, {}):
                    manager.add_translation(
                        message_key, 
                        result, 
                        SupportedLanguage.ENGLISH,
                        category
                    )
                
                # Return translated version
                return manager.translate(message_key)
            
            return result
        
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    # Initialize internationalization
    i18n = get_i18n_manager()
    
    print("🌍 Internationalization Testing")
    print("=" * 40)
    
    # Test basic translation
    print("\\nBasic Translation Test:")
    for lang in [SupportedLanguage.ENGLISH, SupportedLanguage.SPANISH, SupportedLanguage.JAPANESE]:
        msg = i18n.translate("optimization_started", lang)
        print(f"{lang.value}: {msg}")
    
    # Test number formatting
    print("\\nNumber Formatting Test:")
    test_number = 1234567.89
    for lang in [SupportedLanguage.ENGLISH, SupportedLanguage.GERMAN, SupportedLanguage.CHINESE_SIMPLIFIED]:
        formatted = i18n.format_number(test_number, lang)
        print(f"{lang.value}: {formatted}")
    
    # Test translation coverage
    print("\\nTranslation Coverage:")
    for lang in [SupportedLanguage.ENGLISH, SupportedLanguage.SPANISH, SupportedLanguage.JAPANESE]:
        coverage = i18n.get_translation_coverage(lang)
        print(f"{lang.value}: {coverage['coverage']:.1f}% coverage ({coverage['translated_keys']}/{coverage['total_keys']} keys)")
    
    # Test translation stats
    print("\\nTranslation Statistics:")
    stats = i18n.get_translation_stats()
    print(f"Total languages supported: {stats['supported_languages']}")
    print(f"Languages with translations: {stats['languages_with_translations']}")
    print(f"Total translation keys: {stats['total_translation_keys']}")
    
    print("\\n✅ Internationalization testing completed!")