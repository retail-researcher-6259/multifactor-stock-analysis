#!/usr/bin/env python3
"""
Configuration loader for automation system
"""

import json
from pathlib import Path
from typing import Dict, Any

AUTOMATION_DIR = Path(__file__).parent
CONFIG_FILE = AUTOMATION_DIR / 'config.json'

def load_config() -> Dict[str, Any]:
    """Load configuration from config.json"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    else:
        # Return default config if file doesn't exist
        return {
            "regime_detection": {
                "schedule": {
                    "enabled": True,
                    "time": "08:00",
                    "timezone": "Asia/Taipei",  # Default for GMT+8
                    "run_on_weekends": False,
                    "check_market_open": True
                }
            }
        }

def get_schedule_config() -> Dict[str, Any]:
    """Get schedule configuration"""
    config = load_config()
    return config.get('regime_detection', {}).get('schedule', {})

def save_config(config: Dict[str, Any]):
    """Save configuration to config.json"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

# Quick update function to change schedule time
def update_schedule_time(time: str, timezone: str = None):
    """Quick function to update schedule time"""
    config = load_config()
    config['regime_detection']['schedule']['time'] = time
    if timezone:
        config['regime_detection']['schedule']['timezone'] = timezone
    save_config(config)
    print(f"Updated schedule: {time} {timezone or config['regime_detection']['schedule']['timezone']}")