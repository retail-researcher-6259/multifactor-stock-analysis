#!/usr/bin/env python3
"""
Setup script for regime detection automation
Configures scheduled tasks on Windows or cron jobs on Linux/Mac
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
import json

AUTOMATION_DIR = Path(__file__).parent
PROJECT_ROOT = AUTOMATION_DIR.parent


class AutomationSetup:
    """Setup automation for different operating systems"""

    def __init__(self):
        self.os_type = platform.system()
        self.scheduler_script = AUTOMATION_DIR / 'regime_detection_scheduler.py'
        self.daemon_script = AUTOMATION_DIR / 'regime_detection_daemon.py'
        self.python_exe = sys.executable

    def setup_windows_task(self):
        """Setup Windows Task Scheduler task"""
        print("Setting up Windows Task Scheduler...")

        task_name = "MSAS_RegimeDetection"

        # Create the task XML
        task_xml = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>MSAS Market Regime Detection - Runs daily at 8AM if market was open yesterday</Description>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>2024-01-01T08:00:00</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByDay>
        <DaysInterval>1</DaysInterval>
      </ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT2H</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{self.python_exe}</Command>
      <Arguments>"{self.scheduler_script}"</Arguments>
      <WorkingDirectory>{PROJECT_ROOT}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>"""

        # Save XML to temp file
        temp_xml = AUTOMATION_DIR / 'temp_task.xml'
        with open(temp_xml, 'w', encoding='utf-16') as f:
            f.write(task_xml)

        try:
            # Delete existing task if it exists
            subprocess.run(['schtasks', '/delete', '/tn', task_name, '/f'],
                           capture_output=True)

            # Create new task
            result = subprocess.run(
                ['schtasks', '/create', '/tn', task_name, '/xml', str(temp_xml)],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print(f" Windows Task '{task_name}' created successfully")
                print(f"  The task will run daily at 8:00 AM")
                print(f"  To view/modify: Open Task Scheduler and look for '{task_name}'")
                print(f"  To run manually: schtasks /run /tn {task_name}")
                print(f"  To delete: schtasks /delete /tn {task_name}")
            else:
                print(f" Failed to create task: {result.stderr}")

        finally:
            # Clean up temp file
            if temp_xml.exists():
                temp_xml.unlink()

    def setup_linux_cron(self):
        """Setup Linux/Mac cron job"""
        print("Setting up cron job...")

        # Create the cron command
        cron_command = f'0 8 * * * cd "{PROJECT_ROOT}" && "{self.python_exe}" "{self.scheduler_script}" >> "{AUTOMATION_DIR}/logs/cron.log" 2>&1'

        # Check if cron job already exists
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)

        if result.returncode == 0:
            existing_crons = result.stdout
        else:
            existing_crons = ""

        if str(self.scheduler_script) in existing_crons:
            print(" Cron job already exists")
            return

        # Add new cron job
        new_crons = existing_crons.rstrip() + '\n' + cron_command + '\n'

        try:
            process = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE, text=True)
            process.communicate(input=new_crons)

            if process.returncode == 0:
                print(" Cron job added successfully")
                print(f"  Command: {cron_command}")
                print("  To view: crontab -l")
                print("  To edit: crontab -e")
                print("  To remove: crontab -r")
            else:
                print(" Failed to add cron job")

        except Exception as e:
            print(f" Error setting up cron: {e}")

    def create_run_scripts(self):
        """Create convenient run scripts"""
        print("Creating run scripts...")

        # Create logs directory
        logs_dir = AUTOMATION_DIR / 'logs'
        logs_dir.mkdir(exist_ok=True)

        # Create status directory
        status_dir = AUTOMATION_DIR / 'status'
        status_dir.mkdir(exist_ok=True)

        if self.os_type == "Windows":
            # Create batch file for Windows
            batch_file = AUTOMATION_DIR / 'run_regime_detection.bat'
            with open(batch_file, 'w') as f:
                f.write(f'@echo off\n')
                f.write(f'cd /d "{PROJECT_ROOT}"\n')
                f.write(f'"{self.python_exe}" "{self.scheduler_script}"\n')
                f.write(f'pause\n')
            print(f"   Created: {batch_file}")

            # Create daemon batch file
            daemon_batch = AUTOMATION_DIR / 'run_daemon.bat'
            with open(daemon_batch, 'w') as f:
                f.write(f'@echo off\n')
                f.write(f'echo Starting Regime Detection Daemon...\n')
                f.write(f'cd /d "{PROJECT_ROOT}"\n')
                f.write(f'"{self.python_exe}" "{self.daemon_script}"\n')
                f.write(f'pause\n')
            print(f"   Created: {daemon_batch}")

        else:
            # Create shell script for Linux/Mac
            shell_file = AUTOMATION_DIR / 'run_regime_detection.sh'
            with open(shell_file, 'w') as f:
                f.write(f'#!/bin/bash\n')
                f.write(f'cd "{PROJECT_ROOT}"\n')
                f.write(f'"{self.python_exe}" "{self.scheduler_script}"\n')

            # Make executable
            shell_file.chmod(0o755)
            print(f"   Created: {shell_file}")

            # Create daemon shell script
            daemon_shell = AUTOMATION_DIR / 'run_daemon.sh'
            with open(daemon_shell, 'w') as f:
                f.write(f'#!/bin/bash\n')
                f.write(f'echo "Starting Regime Detection Daemon..."\n')
                f.write(f'cd "{PROJECT_ROOT}"\n')
                f.write(f'"{self.python_exe}" "{self.daemon_script}"\n')

            daemon_shell.chmod(0o755)
            print(f"   Created: {daemon_shell}")

    def test_automation(self):
        """Test if automation scripts work"""
        print("\nTesting automation scripts...")

        try:
            # Test import
            sys.path.insert(0, str(AUTOMATION_DIR))
            from regime_detection_scheduler import RegimeDetectionAutomation

            print("   Import successful")

            # Test instantiation
            automation = RegimeDetectionAutomation()
            print("   Automation object created")

            # Check prerequisites
            if automation.check_prerequisites():
                print("   All prerequisites met")
            else:
                print("   Some prerequisites missing")

        except Exception as e:
            print(f"   Test failed: {e}")
            return False

        return True

    def install_requirements(self):
        """Install required packages"""
        print("Checking required packages...")

        required_packages = [
            'pandas',
            'pandas_market_calendars',
            'schedule',
            'pytz'
        ]

        missing_packages = []

        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"   {package} installed")
            except ImportError:
                print(f"   {package} missing")
                missing_packages.append(package)

        if missing_packages:
            print(f"\nInstalling missing packages: {', '.join(missing_packages)}")
            subprocess.run([self.python_exe, '-m', 'pip', 'install'] + missing_packages)

    def setup(self):
        """Run full setup"""
        print("=" * 60)
        print("MSAS REGIME DETECTION AUTOMATION SETUP")
        print("=" * 60)
        print(f"Operating System: {self.os_type}")
        print(f"Python: {self.python_exe}")
        print(f"Project Root: {PROJECT_ROOT}")
        print()

        # Install requirements
        self.install_requirements()
        print()

        # Create run scripts
        self.create_run_scripts()
        print()

        # Test automation
        if not self.test_automation():
            print("\n Tests failed. Please fix issues before setting up scheduling.")
            return False
        print()

        # Setup scheduled task based on OS
        if self.os_type == "Windows":
            self.setup_windows_task()
        elif self.os_type in ["Linux", "Darwin"]:  # Darwin is macOS
            self.setup_linux_cron()
        else:
            print(f" Unsupported operating system: {self.os_type}")
            return False

        print("\n" + "=" * 60)
        print("SETUP COMPLETE")
        print("=" * 60)
        print("\nAutomation Options:")
        print("1. Automatic: The scheduled task/cron job will run daily at 8AM")
        print("2. Manual: Run the scripts in the automation folder:")
        print(f"   - One-time run: run_regime_detection.{'bat' if self.os_type == 'Windows' else 'sh'}")
        print(f"   - Daemon mode: run_daemon.{'bat' if self.os_type == 'Windows' else 'sh'}")
        print("3. Python: Direct execution:")
        print(f"   - python automation/regime_detection_scheduler.py")
        print(f"   - python automation/regime_detection_daemon.py")

        return True


def main():
    """Main entry point"""
    setup = AutomationSetup()
    success = setup.setup()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()