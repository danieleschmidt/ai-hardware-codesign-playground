"""
Internationalization (i18n) support for AI Hardware Co-Design Playground.

This module provides multi-language support and localization features
for global deployment compliance.
"""

import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from threading import Lock

from ..utils.logging import get_logger

logger = get_logger(__name__)


class I18nManager:
    """Internationalization manager for the platform."""
    
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'es': 'Spanish (Español)',
        'fr': 'French (Français)', 
        'de': 'German (Deutsch)',
        'ja': 'Japanese (日本語)',
        'zh': 'Chinese (中文)',
        'pt': 'Portuguese (Português)',
        'it': 'Italian (Italiano)',
        'ru': 'Russian (Русский)',
        'ko': 'Korean (한국어)'
    }
    
    def __init__(self, default_language: str = 'en', translations_dir: Optional[str] = None):
        """
        Initialize i18n manager.
        
        Args:
            default_language: Default language code
            translations_dir: Directory containing translation files
        """
        self.default_language = default_language
        self.current_language = default_language
        
        # Set translations directory
        if translations_dir:
            self.translations_dir = Path(translations_dir)
        else:
            self.translations_dir = Path(__file__).parent / 'translations'
        
        # Translation cache
        self._translations: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        
        # Load default language
        self._load_language(default_language)
        
        logger.info(
            "Initialized I18nManager",
            default_language=default_language,
            translations_dir=str(self.translations_dir)
        )
    
    def set_language(self, language_code: str) -> bool:
        """
        Set current language.
        
        Args:
            language_code: Language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            True if language was set successfully
        """
        if language_code not in self.SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language code: {language_code}")
            return False
        
        if self._load_language(language_code):
            self.current_language = language_code
            logger.info(f"Language set to: {self.SUPPORTED_LANGUAGES[language_code]}")
            return True
        
        return False
    
    def get_text(self, key: str, language: Optional[str] = None, **kwargs) -> str:
        """
        Get translated text for a key.
        
        Args:
            key: Translation key (supports dot notation like 'errors.validation.required')
            language: Language code (uses current language if None)
            **kwargs: Variables for string formatting
            
        Returns:
            Translated text or key if translation not found
        """
        lang = language or self.current_language
        
        # Ensure language is loaded
        if lang not in self._translations:
            if not self._load_language(lang):
                lang = self.default_language
        
        # Get translation
        translation = self._get_nested_value(self._translations.get(lang, {}), key)
        
        # Fallback to default language if not found
        if translation is None and lang != self.default_language:
            translation = self._get_nested_value(
                self._translations.get(self.default_language, {}), 
                key
            )
        
        # Final fallback to key itself
        if translation is None:
            logger.warning(f"Translation not found for key: {key} (language: {lang})")
            translation = key
        
        # Format with variables if provided
        if kwargs and isinstance(translation, str):
            try:
                translation = translation.format(**kwargs)
            except (KeyError, ValueError) as e:
                logger.warning(f"Error formatting translation '{key}': {e}")
        
        return translation
    
    def get_language_options(self) -> List[Dict[str, str]]:
        """Get available language options."""
        return [
            {"code": code, "name": name}
            for code, name in self.SUPPORTED_LANGUAGES.items()
        ]
    
    def get_current_language(self) -> Dict[str, str]:
        """Get current language information."""
        return {
            "code": self.current_language,
            "name": self.SUPPORTED_LANGUAGES[self.current_language]
        }
    
    def _load_language(self, language_code: str) -> bool:
        """Load translation file for a language."""
        with self._lock:
            if language_code in self._translations:
                return True
            
            translation_file = self.translations_dir / f"{language_code}.json"
            
            try:
                if translation_file.exists():
                    with open(translation_file, 'r', encoding='utf-8') as f:
                        translations = json.load(f)
                    self._translations[language_code] = translations
                    logger.debug(f"Loaded translations for language: {language_code}")
                    return True
                else:
                    logger.warning(f"Translation file not found: {translation_file}")
                    # Create default translations if this is the default language
                    if language_code == self.default_language:
                        self._create_default_translations(language_code)
                        return True
                    return False
                    
            except Exception as e:
                logger.error(f"Error loading translations for {language_code}: {e}")
                return False
    
    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Optional[str]:
        """Get value from nested dictionary using dot notation."""
        keys = key.split('.')
        current = data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        
        return current if isinstance(current, str) else None
    
    def _create_default_translations(self, language_code: str) -> None:
        """Create default translation file."""
        default_translations = {
            "app": {
                "name": "AI Hardware Co-Design Playground",
                "description": "Interactive environment for co-optimizing neural networks and hardware accelerators"
            },
            "common": {
                "loading": "Loading...",
                "error": "Error",
                "success": "Success",
                "warning": "Warning",
                "info": "Information",
                "cancel": "Cancel",
                "confirm": "Confirm",
                "save": "Save",
                "delete": "Delete",
                "edit": "Edit",
                "create": "Create",
                "update": "Update",
                "close": "Close",
                "back": "Back",
                "next": "Next",
                "previous": "Previous",
                "finish": "Finish"
            },
            "errors": {
                "validation": {
                    "required": "This field is required",
                    "invalid_format": "Invalid format", 
                    "too_long": "Value is too long",
                    "too_short": "Value is too short",
                    "out_of_range": "Value is out of range"
                },
                "server": {
                    "internal_error": "Internal server error",
                    "not_found": "Resource not found",
                    "unauthorized": "Unauthorized access",
                    "forbidden": "Access forbidden",
                    "timeout": "Request timeout"
                },
                "model": {
                    "invalid_format": "Invalid model format",
                    "unsupported_framework": "Unsupported framework",
                    "profile_failed": "Model profiling failed"
                },
                "hardware": {
                    "invalid_config": "Invalid hardware configuration",
                    "design_failed": "Hardware design failed",
                    "simulation_error": "Simulation error"
                }
            },
            "api": {
                "responses": {
                    "model_profile_success": "Model profiled successfully",
                    "accelerator_design_success": "Accelerator designed successfully",
                    "optimization_started": "Optimization job started",
                    "exploration_started": "Design space exploration started",
                    "workflow_created": "Workflow created successfully"
                }
            },
            "cli": {
                "commands": {
                    "verify": {
                        "description": "Verify installation and dependencies",
                        "success": "Installation verified successfully",
                        "failed": "Verification failed"
                    },
                    "profile": {
                        "description": "Profile a neural network model",
                        "progress": "Analyzing model...",
                        "success": "Model profiled successfully"
                    },
                    "design": {
                        "description": "Design a hardware accelerator",
                        "progress": "Designing accelerator...",
                        "success": "Accelerator designed successfully"
                    },
                    "explore": {
                        "description": "Explore design space",
                        "progress": "Exploring {num_samples} design points...",
                        "success": "Exploration completed successfully"
                    }
                }
            },
            "compliance": {
                "gdpr": {
                    "data_processing_notice": "We process your data in accordance with GDPR regulations",
                    "consent_required": "Your consent is required for data processing",
                    "data_retention": "Data will be retained for the minimum necessary period"
                },
                "ccpa": {
                    "privacy_notice": "California residents have specific privacy rights under CCPA",
                    "do_not_sell": "We do not sell your personal information"
                },
                "general": {
                    "privacy_policy": "Privacy Policy",
                    "terms_of_service": "Terms of Service",
                    "cookie_notice": "This site uses cookies to improve your experience"
                }
            }
        }
        
        # Save to file
        translation_file = self.translations_dir / f"{language_code}.json"
        translation_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(translation_file, 'w', encoding='utf-8') as f:
            json.dump(default_translations, f, indent=2, ensure_ascii=False)
        
        self._translations[language_code] = default_translations
        logger.info(f"Created default translations for {language_code}")


# Global i18n manager instance
_i18n_manager = None


def get_i18n_manager() -> I18nManager:
    """Get global i18n manager instance."""
    global _i18n_manager
    
    if _i18n_manager is None:
        _i18n_manager = I18nManager()
    
    return _i18n_manager


def set_language(language_code: str) -> bool:
    """Set current language globally."""
    return get_i18n_manager().set_language(language_code)


def _(key: str, language: Optional[str] = None, **kwargs) -> str:
    """
    Shorthand function for getting translated text.
    
    Args:
        key: Translation key
        language: Optional language override
        **kwargs: Variables for string formatting
        
    Returns:
        Translated text
    """
    return get_i18n_manager().get_text(key, language, **kwargs)


def detect_language_from_request(request) -> str:
    """
    Detect language from HTTP request.
    
    Args:
        request: HTTP request object
        
    Returns:
        Detected language code
    """
    # Check Accept-Language header
    accept_language = getattr(request.headers, 'accept-language', '')
    
    if accept_language:
        # Parse Accept-Language header
        languages = []
        for lang_range in accept_language.split(','):
            lang_range = lang_range.strip()
            if ';' in lang_range:
                lang_code, quality = lang_range.split(';', 1)
                try:
                    q_value = float(quality.split('=')[1])
                except (ValueError, IndexError):
                    q_value = 1.0
            else:
                lang_code = lang_range
                q_value = 1.0
            
            # Extract primary language code
            lang_code = lang_code.strip().split('-')[0].lower()
            
            if lang_code in I18nManager.SUPPORTED_LANGUAGES:
                languages.append((lang_code, q_value))
        
        # Sort by quality value
        languages.sort(key=lambda x: x[1], reverse=True)
        
        if languages:
            return languages[0][0]
    
    # Default fallback
    return 'en'


class ComplianceManager:
    """Manager for compliance with global regulations."""
    
    REGIONS = {
        'EU': {
            'name': 'European Union',
            'regulations': ['GDPR'],
            'data_residency': True,
            'consent_required': True
        },
        'US': {
            'name': 'United States',
            'regulations': ['CCPA', 'COPPA'],
            'data_residency': False,
            'consent_required': True
        },
        'APAC': {
            'name': 'Asia Pacific',
            'regulations': ['PDPA'],
            'data_residency': True,
            'consent_required': True
        },
        'GLOBAL': {
            'name': 'Global',
            'regulations': ['ISO27001', 'SOC2'],
            'data_residency': False,
            'consent_required': False
        }
    }
    
    def __init__(self, region: str = 'GLOBAL'):
        """
        Initialize compliance manager.
        
        Args:
            region: Deployment region
        """
        self.region = region
        self.config = self.REGIONS.get(region, self.REGIONS['GLOBAL'])
        
        logger.info(f"Initialized ComplianceManager for region: {region}")
    
    def get_compliance_info(self) -> Dict[str, Any]:
        """Get compliance information for current region."""
        return {
            'region': self.region,
            'region_name': self.config['name'],
            'regulations': self.config['regulations'],
            'data_residency_required': self.config['data_residency'],
            'consent_required': self.config['consent_required']
        }
    
    def get_privacy_notice(self, language: str = 'en') -> str:
        """Get privacy notice for the region."""
        i18n = get_i18n_manager()
        
        if 'GDPR' in self.config['regulations']:
            return i18n.get_text('compliance.gdpr.data_processing_notice', language)
        elif 'CCPA' in self.config['regulations']:
            return i18n.get_text('compliance.ccpa.privacy_notice', language)
        else:
            return i18n.get_text('compliance.general.privacy_policy', language)
    
    def requires_consent(self) -> bool:
        """Check if user consent is required in this region."""
        return self.config['consent_required']
    
    def requires_data_residency(self) -> bool:
        """Check if data residency is required in this region."""
        return self.config['data_residency']


# Global compliance manager
_compliance_manager = None


def get_compliance_manager() -> ComplianceManager:
    """Get global compliance manager instance."""
    global _compliance_manager
    
    if _compliance_manager is None:
        # Detect region from environment
        region = os.environ.get('COMPLIANCE_REGION', 'GLOBAL')
        _compliance_manager = ComplianceManager(region)
    
    return _compliance_manager