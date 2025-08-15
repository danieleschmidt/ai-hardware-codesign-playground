"""
Internationalization (i18n) support for AI Hardware Co-Design Playground.

Provides multi-language support, localization, and cultural adaptations
for global deployment.
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages for the platform."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    RUSSIAN = "ru"


@dataclass
class LocalizationConfig:
    """Configuration for localization settings."""
    
    language: str
    region: str
    currency: str
    date_format: str
    number_format: str
    decimal_separator: str
    thousands_separator: str
    rtl: bool = False  # Right-to-left text direction


class I18nManager:
    """Internationalization manager for the platform."""
    
    def __init__(self, default_language: str = "en"):
        """
        Initialize i18n manager.
        
        Args:
            default_language: Default language code
        """
        self.default_language = default_language
        self.current_language = default_language
        self._translations = {}
        self._localization_configs = {}
        
        # Load default translations and configs
        self._load_default_translations()
        self._load_localization_configs()
        
        logger.info(f"Initialized I18nManager with default language: {default_language}")
    
    def set_language(self, language: str) -> bool:
        """
        Set current language.
        
        Args:
            language: Language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            True if language was set successfully
        """
        if language in self._translations:
            self.current_language = language
            logger.info(f"Language set to: {language}")
            return True
        else:
            logger.warning(f"Language not supported: {language}")
            return False
    
    def translate(self, key: str, language: Optional[str] = None, **kwargs) -> str:
        """
        Translate a text key to the specified or current language.
        
        Args:
            key: Translation key
            language: Target language (defaults to current)
            **kwargs: Variables for string formatting
            
        Returns:
            Translated text
        """
        lang = language or self.current_language
        
        # Get translation from language-specific dict
        lang_dict = self._translations.get(lang, {})
        text = lang_dict.get(key)
        
        # Fallback to default language
        if text is None and lang != self.default_language:
            default_dict = self._translations.get(self.default_language, {})
            text = default_dict.get(key)
        
        # Fallback to key itself
        if text is None:
            text = key
            logger.warning(f"Missing translation for key '{key}' in language '{lang}'")
        
        # Format with variables if provided
        if kwargs:
            try:
                text = text.format(**kwargs)
            except (KeyError, ValueError) as e:
                logger.error(f"Error formatting translation '{key}': {e}")
        
        return text
    
    def get_localization_config(self, language: Optional[str] = None) -> LocalizationConfig:
        """
        Get localization configuration for a language.
        
        Args:
            language: Language code (defaults to current)
            
        Returns:
            Localization configuration
        """
        lang = language or self.current_language
        return self._localization_configs.get(lang, self._localization_configs[self.default_language])
    
    def format_number(self, number: float, language: Optional[str] = None) -> str:
        """
        Format number according to locale.
        
        Args:
            number: Number to format
            language: Language code (defaults to current)
            
        Returns:
            Formatted number string
        """
        config = self.get_localization_config(language)
        
        # Simple number formatting
        if isinstance(number, int):
            return f"{number:,}".replace(",", config.thousands_separator)
        else:
            formatted = f"{number:,.2f}"
            formatted = formatted.replace(",", "|").replace(".", config.decimal_separator)
            formatted = formatted.replace("|", config.thousands_separator)
            return formatted
    
    def format_currency(self, amount: float, language: Optional[str] = None) -> str:
        """
        Format currency according to locale.
        
        Args:
            amount: Currency amount
            language: Language code (defaults to current)
            
        Returns:
            Formatted currency string
        """
        config = self.get_localization_config(language)
        formatted_number = self.format_number(amount, language)
        
        # Currency symbol placement varies by locale
        if config.language in ["en"]:
            return f"${formatted_number}"
        elif config.language in ["es", "fr", "it", "pt"]:
            return f"{formatted_number} €"
        elif config.language in ["de"]:
            return f"{formatted_number} €"
        elif config.language in ["ja"]:
            return f"¥{formatted_number}"
        elif config.language in ["zh"]:
            return f"¥{formatted_number}"
        else:
            return f"{config.currency} {formatted_number}"
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """
        Get list of supported languages.
        
        Returns:
            List of language info dictionaries
        """
        return [
            {"code": lang, "name": self.translate("language_name", lang)}
            for lang in self._translations.keys()
        ]
    
    def _load_default_translations(self) -> None:
        """Load default translation dictionaries."""
        
        # English (default)
        self._translations["en"] = {
            "language_name": "English",
            "welcome": "Welcome to AI Hardware Co-Design Playground",
            "model_profile": "Model Profile",
            "accelerator_design": "Accelerator Design",
            "optimization": "Optimization",
            "simulation": "Simulation",
            "results": "Results",
            "error": "Error",
            "success": "Success",
            "warning": "Warning",
            "info": "Information",
            "performance": "Performance",
            "power_consumption": "Power Consumption",
            "throughput": "Throughput",
            "latency": "Latency",
            "accuracy": "Accuracy",
            "compute_units": "Compute Units",
            "memory_bandwidth": "Memory Bandwidth",
            "frequency": "Frequency",
            "dataflow": "Dataflow",
            "precision": "Precision",
            "design_space_exploration": "Design Space Exploration",
            "pareto_frontier": "Pareto Frontier",
            "optimization_target": "Optimization Target",
            "hardware_generation": "Hardware Generation",
            "rtl_output": "RTL Output",
            "validation_error": "Validation Error: {message}",
            "processing": "Processing...",
            "completed": "Completed",
            "failed": "Failed",
            "fps": "FPS (Frames Per Second)",
            "power_mw": "Power (mW)",
            "area_mm2": "Area (mm²)",
            "energy_efficiency": "Energy Efficiency (TOPS/W)",
        }
        
        # Spanish
        self._translations["es"] = {
            "language_name": "Español",
            "welcome": "Bienvenido al Playground de Co-diseño de Hardware de IA",
            "model_profile": "Perfil del Modelo",
            "accelerator_design": "Diseño del Acelerador",
            "optimization": "Optimización",
            "simulation": "Simulación",
            "results": "Resultados",
            "error": "Error",
            "success": "Éxito",
            "warning": "Advertencia",
            "info": "Información",
            "performance": "Rendimiento",
            "power_consumption": "Consumo de Energía",
            "throughput": "Rendimiento",
            "latency": "Latencia",
            "accuracy": "Precisión",
            "compute_units": "Unidades de Cómputo",
            "memory_bandwidth": "Ancho de Banda de Memoria",
            "frequency": "Frecuencia",
            "dataflow": "Flujo de Datos",
            "precision": "Precisión",
            "design_space_exploration": "Exploración del Espacio de Diseño",
            "pareto_frontier": "Frontera de Pareto",
            "optimization_target": "Objetivo de Optimización",
            "hardware_generation": "Generación de Hardware",
            "rtl_output": "Salida RTL",
            "validation_error": "Error de Validación: {message}",
            "processing": "Procesando...",
            "completed": "Completado",
            "failed": "Falló",
            "fps": "FPS (Fotogramas Por Segundo)",
            "power_mw": "Potencia (mW)",
            "area_mm2": "Área (mm²)",
            "energy_efficiency": "Eficiencia Energética (TOPS/W)",
        }
        
        # French
        self._translations["fr"] = {
            "language_name": "Français",
            "welcome": "Bienvenue dans le Playground de Co-conception Hardware IA",
            "model_profile": "Profil du Modèle",
            "accelerator_design": "Conception d'Accélérateur",
            "optimization": "Optimisation",
            "simulation": "Simulation",
            "results": "Résultats",
            "error": "Erreur",
            "success": "Succès",
            "warning": "Avertissement",
            "info": "Information",
            "performance": "Performance",
            "power_consumption": "Consommation d'Énergie",
            "throughput": "Débit",
            "latency": "Latence",
            "accuracy": "Précision",
            "compute_units": "Unités de Calcul",
            "memory_bandwidth": "Bande Passante Mémoire",
            "frequency": "Fréquence",
            "dataflow": "Flux de Données",
            "precision": "Précision",
            "design_space_exploration": "Exploration de l'Espace de Conception",
            "pareto_frontier": "Frontière de Pareto",
            "optimization_target": "Cible d'Optimisation",
            "hardware_generation": "Génération de Matériel",
            "rtl_output": "Sortie RTL",
            "validation_error": "Erreur de Validation: {message}",
            "processing": "Traitement...",
            "completed": "Terminé",
            "failed": "Échoué",
            "fps": "IPS (Images Par Seconde)",
            "power_mw": "Puissance (mW)",
            "area_mm2": "Surface (mm²)",
            "energy_efficiency": "Efficacité Énergétique (TOPS/W)",
        }
        
        # German
        self._translations["de"] = {
            "language_name": "Deutsch",
            "welcome": "Willkommen im KI-Hardware-Co-Design-Spielplatz",
            "model_profile": "Modellprofil",
            "accelerator_design": "Beschleuniger-Design",
            "optimization": "Optimierung",
            "simulation": "Simulation",
            "results": "Ergebnisse",
            "error": "Fehler",
            "success": "Erfolg",
            "warning": "Warnung",
            "info": "Information",
            "performance": "Leistung",
            "power_consumption": "Stromverbrauch",
            "throughput": "Durchsatz",
            "latency": "Latenz",
            "accuracy": "Genauigkeit",
            "compute_units": "Recheneinheiten",
            "memory_bandwidth": "Speicherbandbreite",
            "frequency": "Frequenz",
            "dataflow": "Datenfluss",
            "precision": "Präzision",
            "design_space_exploration": "Design-Raum-Exploration",
            "pareto_frontier": "Pareto-Grenze",
            "optimization_target": "Optimierungsziel",
            "hardware_generation": "Hardware-Generierung",
            "rtl_output": "RTL-Ausgabe",
            "validation_error": "Validierungsfehler: {message}",
            "processing": "Verarbeitung...",
            "completed": "Abgeschlossen",
            "failed": "Fehlgeschlagen",
            "fps": "BPS (Bilder Pro Sekunde)",
            "power_mw": "Leistung (mW)",
            "area_mm2": "Fläche (mm²)",
            "energy_efficiency": "Energieeffizienz (TOPS/W)",
        }
        
        # Japanese
        self._translations["ja"] = {
            "language_name": "日本語",
            "welcome": "AIハードウェア協調設計プレイグラウンドへようこそ",
            "model_profile": "モデルプロファイル",
            "accelerator_design": "アクセラレータ設計",
            "optimization": "最適化",
            "simulation": "シミュレーション",
            "results": "結果",
            "error": "エラー",
            "success": "成功",
            "warning": "警告",
            "info": "情報",
            "performance": "性能",
            "power_consumption": "消費電力",
            "throughput": "スループット",
            "latency": "レイテンシ",
            "accuracy": "精度",
            "compute_units": "計算ユニット",
            "memory_bandwidth": "メモリ帯域幅",
            "frequency": "周波数",
            "dataflow": "データフロー",
            "precision": "精度",
            "design_space_exploration": "設計空間探索",
            "pareto_frontier": "パレートフロンティア",
            "optimization_target": "最適化目標",
            "hardware_generation": "ハードウェア生成",
            "rtl_output": "RTL出力",
            "validation_error": "検証エラー: {message}",
            "processing": "処理中...",
            "completed": "完了",
            "failed": "失敗",
            "fps": "FPS（フレーム毎秒）",
            "power_mw": "電力 (mW)",
            "area_mm2": "面積 (mm²)",
            "energy_efficiency": "エネルギー効率 (TOPS/W)",
        }
        
        # Chinese Simplified
        self._translations["zh"] = {
            "language_name": "中文",
            "welcome": "欢迎来到AI硬件协同设计平台",
            "model_profile": "模型概况",
            "accelerator_design": "加速器设计",
            "optimization": "优化",
            "simulation": "仿真",
            "results": "结果",
            "error": "错误",
            "success": "成功",
            "warning": "警告",
            "info": "信息",
            "performance": "性能",
            "power_consumption": "功耗",
            "throughput": "吞吐量",
            "latency": "延迟",
            "accuracy": "精度",
            "compute_units": "计算单元",
            "memory_bandwidth": "内存带宽",
            "frequency": "频率",
            "dataflow": "数据流",
            "precision": "精度",
            "design_space_exploration": "设计空间探索",
            "pareto_frontier": "帕累托前沿",
            "optimization_target": "优化目标",
            "hardware_generation": "硬件生成",
            "rtl_output": "RTL输出",
            "validation_error": "验证错误：{message}",
            "processing": "处理中...",
            "completed": "已完成",
            "failed": "失败",
            "fps": "FPS（每秒帧数）",
            "power_mw": "功率 (mW)",
            "area_mm2": "面积 (mm²)",
            "energy_efficiency": "能效 (TOPS/W)",
        }
    
    def _load_localization_configs(self) -> None:
        """Load localization configurations for different regions."""
        
        self._localization_configs = {
            "en": LocalizationConfig(
                language="en", region="US", currency="USD",
                date_format="%m/%d/%Y", number_format="1,234.56",
                decimal_separator=".", thousands_separator=","
            ),
            "es": LocalizationConfig(
                language="es", region="ES", currency="EUR",
                date_format="%d/%m/%Y", number_format="1.234,56",
                decimal_separator=",", thousands_separator="."
            ),
            "fr": LocalizationConfig(
                language="fr", region="FR", currency="EUR",
                date_format="%d/%m/%Y", number_format="1 234,56",
                decimal_separator=",", thousands_separator=" "
            ),
            "de": LocalizationConfig(
                language="de", region="DE", currency="EUR",
                date_format="%d.%m.%Y", number_format="1.234,56",
                decimal_separator=",", thousands_separator="."
            ),
            "ja": LocalizationConfig(
                language="ja", region="JP", currency="JPY",
                date_format="%Y/%m/%d", number_format="1,234",
                decimal_separator=".", thousands_separator=","
            ),
            "zh": LocalizationConfig(
                language="zh", region="CN", currency="CNY",
                date_format="%Y-%m-%d", number_format="1,234.56",
                decimal_separator=".", thousands_separator=","
            ),
        }


# Global i18n manager instance
_i18n_manager: Optional[I18nManager] = None


def get_i18n_manager() -> I18nManager:
    """Get global i18n manager instance."""
    global _i18n_manager
    
    if _i18n_manager is None:
        _i18n_manager = I18nManager()
    
    return _i18n_manager


def translate(key: str, language: Optional[str] = None, **kwargs) -> str:
    """
    Convenience function for translation.
    
    Args:
        key: Translation key
        language: Target language (defaults to current)
        **kwargs: Variables for string formatting
        
    Returns:
        Translated text
    """
    return get_i18n_manager().translate(key, language, **kwargs)


def set_language(language: str) -> bool:
    """
    Convenience function to set current language.
    
    Args:
        language: Language code
        
    Returns:
        True if successful
    """
    return get_i18n_manager().set_language(language)


def format_number(number: float, language: Optional[str] = None) -> str:
    """
    Convenience function for number formatting.
    
    Args:
        number: Number to format
        language: Language code (defaults to current)
        
    Returns:
        Formatted number string
    """
    return get_i18n_manager().format_number(number, language)


def format_currency(amount: float, language: Optional[str] = None) -> str:
    """
    Convenience function for currency formatting.
    
    Args:
        amount: Currency amount
        language: Language code (defaults to current)
        
    Returns:
        Formatted currency string
    """
    return get_i18n_manager().format_currency(amount, language)