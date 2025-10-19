#!/usr/bin/env python3
"""
Git Safety Check Script
Verifies repository is safe before Phase 3 (embeddings & vector DB)

Checks:
1. No large files (>50MB) are staged
2. No API keys or secrets in staged files
3. No sensitive data files
4. .gitignore is configured correctly
"""

import subprocess
import sys
import re
from pathlib import Path
from typing import List, Tuple

# ANSI colors
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

# File size limit (50 MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

# Patterns to search for secrets
SECRET_PATTERNS = [
    r'api[_-]?key\s*[=:]\s*["\'][^"\']{10,}["\']',
    r'password\s*[=:]\s*["\'][^"\']+["\']',
    r'secret\s*[=:]\s*["\'][^"\']{10,}["\']',
    r'token\s*[=:]\s*["\'][^"\']{10,}["\']',
    r'bearer\s+[a-zA-Z0-9_\-\.]{20,}',
    r'aws_access_key_id',
    r'aws_secret_access_key',
    r'AKIA[0-9A-Z]{16}',  # AWS Access Key
]

# File extensions that shouldn't be committed
FORBIDDEN_EXTENSIONS = [
    '.nc', '.grib', '.grib2', '.hdf5', '.h5',  # Climate data
    '.pkl', '.pickle',  # Pickled objects (can be large)
    '.bin', '.dat',  # Binary data
]


def run_command(cmd: str) -> Tuple[int, str]:
    """Run shell command and return exit code and output."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    return result.returncode, result.stdout + result.stderr


def check_large_files() -> List[str]:
    """Check for large files in git."""
    print(f"{BLUE}🔍 Checking for large files...{RESET}")
    
    issues = []
    
    # Check staged files
    code, output = run_command("git ls-files --stage")
    if code != 0:
        return [f"Could not check git files: {output}"]
    
    for line in output.strip().split('\n'):
        if not line:
            continue
        
        parts = line.split()
        if len(parts) >= 4:
            size = int(parts[3])
            filepath = ' '.join(parts[4:])
            
            if size > MAX_FILE_SIZE:
                size_mb = size / (1024 * 1024)
                issues.append(
                    f"Large file: {filepath} ({size_mb:.2f} MB)"
                )
    
    if issues:
        print(f"{RED}❌ Found {len(issues)} large files{RESET}")
    else:
        print(f"{GREEN}✅ No large files found{RESET}")
    
    return issues


def check_secrets() -> List[str]:
    """Check for secrets/API keys in git."""
    print(f"{BLUE}🔍 Checking for secrets...{RESET}")
    
    issues = []
    
    for pattern in SECRET_PATTERNS:
        code, output = run_command(f'git grep -E "{pattern}" --cached')
        
        if code == 0 and output.strip():
            lines = output.strip().split('\n')
            for line in lines:
                # Skip false positives in .gitignore and this script
                if '.gitignore' in line or 'git_safety_check.py' in line:
                    continue
                issues.append(f"Possible secret: {line[:100]}")
    
    if issues:
        print(f"{RED}❌ Found {len(issues)} potential secrets{RESET}")
    else:
        print(f"{GREEN}✅ No secrets detected{RESET}")
    
    return issues


def check_forbidden_files() -> List[str]:
    """Check for forbidden file extensions."""
    print(f"{BLUE}🔍 Checking for forbidden file types...{RESET}")
    
    issues = []
    
    code, output = run_command("git ls-files")
    if code != 0:
        return [f"Could not list git files: {output}"]
    
    for filepath in output.strip().split('\n'):
        if not filepath:
            continue
        
        ext = Path(filepath).suffix.lower()
        if ext in FORBIDDEN_EXTENSIONS:
            issues.append(f"Forbidden file type: {filepath}")
    
    if issues:
        print(f"{RED}❌ Found {len(issues)} forbidden files{RESET}")
    else:
        print(f"{GREEN}✅ No forbidden files found{RESET}")
    
    return issues


def check_gitignore() -> List[str]:
    """Check if .gitignore has critical patterns."""
    print(f"{BLUE}🔍 Checking .gitignore configuration...{RESET}")
    
    issues = []
    gitignore_path = Path('.gitignore')
    
    if not gitignore_path.exists():
        return [".gitignore file not found!"]
    
    content = gitignore_path.read_text()
    
    # Critical patterns that MUST be in .gitignore
    required_patterns = [
        ('*.nc', 'NetCDF climate data'),
        ('data/raw/**', 'Raw data directory'),
        ('data/processed/**', 'Processed data directory'),
        ('.env', 'Environment variables'),
        ('*.log', 'Log files'),
        ('chroma_db/', 'ChromaDB vector storage'),
        ('embeddings/', 'Embeddings cache'),
    ]
    
    for pattern, description in required_patterns:
        if pattern not in content:
            issues.append(f"Missing pattern in .gitignore: {pattern} ({description})")
    
    if issues:
        print(f"{YELLOW}⚠️  .gitignore has {len(issues)} missing patterns{RESET}")
    else:
        print(f"{GREEN}✅ .gitignore is properly configured{RESET}")
    
    return issues


def check_git_status() -> List[str]:
    """Check git status for unexpected files."""
    print(f"{BLUE}🔍 Checking git status...{RESET}")
    
    issues = []
    
    code, output = run_command("git status --porcelain")
    if code != 0:
        return [f"Could not check git status: {output}"]
    
    # Count untracked and modified files
    untracked = sum(1 for line in output.split('\n') if line.startswith('??'))
    modified = sum(1 for line in output.split('\n') if line.startswith(' M'))
    
    if untracked > 50:
        issues.append(f"Warning: {untracked} untracked files (consider adding to .gitignore)")
    
    print(f"{GREEN}✅ Git status: {modified} modified, {untracked} untracked{RESET}")
    
    return issues


def main():
    """Run all safety checks."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}🛡️  Git Safety Check for Phase 3{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    all_issues = []
    
    # Run all checks
    all_issues.extend(check_large_files())
    all_issues.extend(check_secrets())
    all_issues.extend(check_forbidden_files())
    all_issues.extend(check_gitignore())
    all_issues.extend(check_git_status())
    
    print(f"\n{BLUE}{'='*60}{RESET}")
    
    # Summary
    if all_issues:
        print(f"{RED}❌ FAILED: Found {len(all_issues)} issues:{RESET}\n")
        for issue in all_issues:
            print(f"  • {issue}")
        print(f"\n{YELLOW}Please fix these issues before pushing to GitHub!{RESET}")
        sys.exit(1)
    else:
        print(f"{GREEN}✅ SUCCESS: Repository is safe for Phase 3!{RESET}")
        print(f"{GREEN}   - No large files{RESET}")
        print(f"{GREEN}   - No secrets detected{RESET}")
        print(f"{GREEN}   - .gitignore is configured{RESET}")
        print(f"\n{GREEN}🚀 Safe to proceed with embeddings & vector DB!{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
