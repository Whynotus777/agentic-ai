# agent/application_builder.py
"""
Complete application building domain with multi-repo refactoring,
Yi-Coder-9B file system agent, DeepSeek-Coder-6.7B code generation,
sandboxed execution, and HITL approval before commits.
"""

import asyncio
import json
import os
import tempfile
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
import uuid
import ast
import black
import git

from opentelemetry import trace
tracer = trace.get_tracer(__name__)


class CodeTaskType(Enum):
    """Types of code tasks"""
    GENERATE_CODE = "generate_code"
    REFACTOR = "refactor"
    ADD_FEATURE = "add_feature"
    FIX_BUG = "fix_bug"
    OPTIMIZE = "optimize"
    ADD_TESTS = "add_tests"
    DOCUMENTATION = "documentation"
    MULTI_REPO_REFACTOR = "multi_repo_refactor"


class CodeLanguage(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    CSHARP = "csharp"


@dataclass
class CodeFile:
    """Represents a code file"""
    path: str
    content: str
    language: CodeLanguage
    original_content: Optional[str] = None
    changes: List[Dict[str, Any]] = field(default_factory=list)
    test_file: Optional[str] = None


@dataclass
class CodeProject:
    """Represents a code project"""
    project_id: str
    name: str
    repository_url: str
    branch: str
    files: Dict[str, CodeFile]
    dependencies: List[str]
    test_coverage: float = 0.0
    build_status: str = "unknown"


@dataclass
class RefactorPlan:
    """Plan for code refactoring"""
    plan_id: str
    projects: List[CodeProject]
    changes: List[Dict[str, Any]]
    estimated_impact: Dict[str, Any]
    requires_approval: bool
    approval_status: Optional[str] = None


class ApplicationBuildingOrchestrator:
    """
    Orchestrator for application building tasks with multi-repo support
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.projects: Dict[str, CodeProject] = {}
        self.refactor_plans: Dict[str, RefactorPlan] = {}
        self.sandbox_environments: Dict[str, str] = {}
        self.hitl_approvals: Dict[str, bool] = {}
        self.file_system_agent = FileSystemAgent()
        self.code_generation_agent = CodeGenerationAgent()
        
    @tracer.start_as_current_span("plan_multi_repo_refactor")
    async def plan_multi_repo_refactor(
        self,
        repositories: List[str],
        refactor_description: str,
        target_improvements: Dict[str, Any]
    ) -> RefactorPlan:
        """
        Plan a multi-repository refactoring operation
        """
        span = trace.get_current_span()
        
        plan_id = f"refactor-{uuid.uuid4().hex[:12]}"
        projects = []
        
        # Analyze each repository
        for repo_url in repositories:
            project = await self._analyze_repository(repo_url)
            projects.append(project)
            self.projects[project.project_id] = project
        
        # Generate refactoring plan
        changes = await self._generate_refactor_plan(
            projects,
            refactor_description,
            target_improvements
        )
        
        # Estimate impact
        impact = await self._estimate_impact(projects, changes)
        
        # Determine if approval required
        requires_approval = (
            impact.get("files_affected", 0) > 10 or
            impact.get("breaking_changes", False) or
            len(projects) > 1
        )
        
        plan = RefactorPlan(
            plan_id=plan_id,
            projects=projects,
            changes=changes,
            estimated_impact=impact,
            requires_approval=requires_approval
        )
        
        self.refactor_plans[plan_id] = plan
        
        span.set_attributes({
            "plan_id": plan_id,
            "num_repositories": len(repositories),
            "num_changes": len(changes),
            "requires_approval": requires_approval
        })
        
        return plan
    
    async def _analyze_repository(self, repo_url: str) -> CodeProject:
        """Analyze a repository structure"""
        project_id = f"project-{uuid.uuid4().hex[:12]}"
        
        # Clone repository to temp directory
        temp_dir = tempfile.mkdtemp(prefix=f"repo_{project_id}_")
        self.sandbox_environments[project_id] = temp_dir
        
        # Use git to clone
        try:
            repo = git.Repo.clone_from(repo_url, temp_dir)
            branch = repo.active_branch.name
        except Exception as e:
            print(f"Error cloning repository: {e}")
            branch = "main"
        
        # Scan files using file system agent
        files = await self.file_system_agent.scan_directory(temp_dir)
        
        # Extract project info
        project_name = os.path.basename(repo_url).replace(".git", "")
        
        # Detect dependencies
        dependencies = await self._detect_dependencies(temp_dir, files)
        
        project = CodeProject(
            project_id=project_id,
            name=project_name,
            repository_url=repo_url,
            branch=branch,
            files=files,
            dependencies=dependencies
        )
        
        return project
    
    async def _detect_dependencies(
        self,
        project_dir: str,
        files: Dict[str, CodeFile]
    ) -> List[str]:
        """Detect project dependencies"""
        dependencies = []
        
        # Python dependencies
        if "requirements.txt" in files:
            deps = files["requirements.txt"].content.split("\n")
            dependencies.extend([d.strip() for d in deps if d.strip()])
        
        # JavaScript/TypeScript dependencies
        if "package.json" in files:
            try:
                package_json = json.loads(files["package.json"].content)
                dependencies.extend(package_json.get("dependencies", {}).keys())
                dependencies.extend(package_json.get("devDependencies", {}).keys())
            except json.JSONDecodeError:
                pass
        
        return dependencies
    
    async def _generate_refactor_plan(
        self,
        projects: List[CodeProject],
        description: str,
        improvements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate detailed refactoring plan"""
        changes = []
        
        for project in projects:
            # Analyze code quality issues
            issues = await self._analyze_code_quality(project)
            
            # Generate changes based on improvements
            if improvements.get("performance"):
                perf_changes = await self._plan_performance_improvements(project)
                changes.extend(perf_changes)
            
            if improvements.get("security"):
                sec_changes = await self._plan_security_improvements(project)
                changes.extend(sec_changes)
            
            if improvements.get("maintainability"):
                maint_changes = await self._plan_maintainability_improvements(project)
                changes.extend(maint_changes)
            
            if improvements.get("testing"):
                test_changes = await self._plan_test_improvements(project)
                changes.extend(test_changes)
        
        return changes
    
    async def _analyze_code_quality(
        self,
        project: CodeProject
    ) -> List[Dict[str, Any]]:
        """Analyze code quality issues"""
        issues = []
        
        for file_path, code_file in project.files.items():
            if code_file.language == CodeLanguage.PYTHON:
                # Check Python code quality
                try:
                    ast.parse(code_file.content)
                except SyntaxError as e:
                    issues.append({
                        "file": file_path,
                        "issue": "syntax_error",
                        "details": str(e)
                    })
                
                # Check formatting
                try:
                    formatted = black.format_str(code_file.content, mode=black.Mode())
                    if formatted != code_file.content:
                        issues.append({
                            "file": file_path,
                            "issue": "formatting",
                            "details": "Code needs formatting"
                        })
                except Exception:
                    pass
        
        return issues
    
    async def _plan_performance_improvements(
        self,
        project: CodeProject
    ) -> List[Dict[str, Any]]:
        """Plan performance improvements"""
        changes = []
        
        # Analyze for common performance issues
        for file_path, code_file in project.files.items():
            if code_file.language == CodeLanguage.PYTHON:
                content = code_file.content
                
                # Check for inefficient patterns
                if "for i in range(len(" in content:
                    changes.append({
                        "project_id": project.project_id,
                        "file": file_path,
                        "type": "performance",
                        "description": "Replace range(len()) with enumerate()",
                        "priority": "medium"
                    })
                
                if "+" in content and "join(" not in content and "str" in content:
                    changes.append({
                        "project_id": project.project_id,
                        "file": file_path,
                        "type": "performance",
                        "description": "Use join() for string concatenation",
                        "priority": "low"
                    })
        
        return changes
    
    async def _plan_security_improvements(
        self,
        project: CodeProject
    ) -> List[Dict[str, Any]]:
        """Plan security improvements"""
        changes = []
        
        for file_path, code_file in project.files.items():
            content = code_file.content
            
            # Check for security issues
            security_patterns = [
                ("eval(", "Remove eval() usage"),
                ("exec(", "Remove exec() usage"),
                ("pickle.loads", "Avoid pickle for untrusted data"),
                ("os.system", "Use subprocess instead of os.system"),
                ("password =", "Don't hardcode passwords"),
                ("api_key =", "Don't hardcode API keys")
            ]
            
            for pattern, description in security_patterns:
                if pattern in content:
                    changes.append({
                        "project_id": project.project_id,
                        "file": file_path,
                        "type": "security",
                        "description": description,
                        "priority": "high"
                    })
        
        return changes
    
    async def _plan_maintainability_improvements(
        self,
        project: CodeProject
    ) -> List[Dict[str, Any]]:
        """Plan maintainability improvements"""
        changes = []
        
        for file_path, code_file in project.files.items():
            if code_file.language == CodeLanguage.PYTHON:
                # Check for long functions
                lines = code_file.content.split("\n")
                in_function = False
                function_start = 0
                function_name = ""
                
                for i, line in enumerate(lines):
                    if line.strip().startswith("def "):
                        in_function = True
                        function_start = i
                        function_name = line.split("(")[0].replace("def ", "")
                    elif in_function and not line.startswith(" ") and not line.startswith("\t"):
                        function_length = i - function_start
                        if function_length > 50:
                            changes.append({
                                "project_id": project.project_id,
                                "file": file_path,
                                "type": "maintainability",
                                "description": f"Function {function_name} is too long ({function_length} lines)",
                                "priority": "medium"
                            })
                        in_function = False
        
        return changes
    
    async def _plan_test_improvements(
        self,
        project: CodeProject
    ) -> List[Dict[str, Any]]:
        """Plan test improvements"""
        changes = []
        
        # Check test coverage
        test_files = [f for f in project.files if "test" in f.lower()]
        code_files = [f for f in project.files if not "test" in f.lower() and f.endswith(".py")]
        
        if len(test_files) < len(code_files) * 0.5:
            changes.append({
                "project_id": project.project_id,
                "type": "testing",
                "description": "Insufficient test coverage - add more test files",
                "priority": "high"
            })
        
        # Check for missing tests for specific files
        for code_file in code_files:
            test_file = code_file.replace(".py", "_test.py")
            if test_file not in test_files:
                changes.append({
                    "project_id": project.project_id,
                    "file": code_file,
                    "type": "testing",
                    "description": f"Add tests for {code_file}",
                    "priority": "medium"
                })
        
        return changes
    
    async def _estimate_impact(
        self,
        projects: List[CodeProject],
        changes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Estimate impact of changes"""
        files_affected = len(set(c.get("file", "") for c in changes if c.get("file")))
        
        # Check for breaking changes
        breaking_changes = any(
            c.get("type") == "api_change" or
            "breaking" in c.get("description", "").lower()
            for c in changes
        )
        
        # Estimate time
        time_per_change = {
            "performance": 30,
            "security": 45,
            "maintainability": 20,
            "testing": 60
        }
        
        estimated_minutes = sum(
            time_per_change.get(c.get("type", "other"), 15)
            for c in changes
        )
        
        return {
            "files_affected": files_affected,
            "total_changes": len(changes),
            "breaking_changes": breaking_changes,
            "estimated_time_minutes": estimated_minutes,
            "risk_level": "high" if breaking_changes else "medium" if files_affected > 10 else "low"
        }
    
    async def execute_refactor_plan(
        self,
        plan_id: str
    ) -> Dict[str, Any]:
        """Execute a refactoring plan"""
        if plan_id not in self.refactor_plans:
            raise ValueError(f"Plan {plan_id} not found")
        
        plan = self.refactor_plans[plan_id]
        
        # Check HITL approval if required
        if plan.requires_approval and not plan.approval_status == "approved":
            approval_id = await self._request_hitl_approval(plan)
            
            # Wait for approval
            if not self.hitl_approvals.get(approval_id, False):
                return {
                    "status": "waiting_approval",
                    "approval_id": approval_id
                }
        
        results = []
        
        # Execute changes for each project
        for project in plan.projects:
            project_results = await self._execute_project_changes(
                project,
                [c for c in plan.changes if c.get("project_id") == project.project_id]
            )
            results.append(project_results)
        
        return {
            "status": "completed",
            "results": results
        }
    
    async def _execute_project_changes(
        self,
        project: CodeProject,
        changes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute changes for a project"""
        sandbox_dir = self.sandbox_environments[project.project_id]
        modified_files = []
        
        for change in changes:
            if change.get("file"):
                file_path = os.path.join(sandbox_dir, change["file"])
                
                # Apply change based on type
                if change["type"] == "performance":
                    await self._apply_performance_fix(file_path, change)
                elif change["type"] == "security":
                    await self._apply_security_fix(file_path, change)
                elif change["type"] == "maintainability":
                    await self._apply_maintainability_fix(file_path, change)
                elif change["type"] == "testing":
                    await self._generate_tests(file_path, change)
                
                modified_files.append(change["file"])
        
        # Run tests in sandbox
        test_results = await self._run_tests_sandboxed(sandbox_dir)
        
        # Only commit if tests pass
        if test_results["passed"]:
            commit_result = await self._commit_changes(project, modified_files)
            return {
                "project": project.name,
                "status": "success",
                "modified_files": modified_files,
                "test_results": test_results,
                "commit": commit_result
            }
        else:
            return {
                "project": project.name,
                "status": "tests_failed",
                "modified_files": modified_files,
                "test_results": test_results
            }
    
    async def _apply_performance_fix(
        self,
        file_path: str,
        change: Dict[str, Any]
    ):
        """Apply performance optimization"""
        # Use code generation agent to fix performance issues
        with open(file_path, 'r') as f:
            original_content = f.read()
        
        fixed_content = await self.code_generation_agent.optimize_code(
            original_content,
            change["description"]
        )
        
        with open(file_path, 'w') as f:
            f.write(fixed_content)
    
    async def _apply_security_fix(
        self,
        file_path: str,
        change: Dict[str, Any]
    ):
        """Apply security fix"""
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Simple pattern replacements for common security issues
        replacements = {
            "eval(": "ast.literal_eval(",
            "os.system": "subprocess.run",
            "pickle.loads": "json.loads"
        }
        
        for pattern, replacement in replacements.items():
            if pattern in content:
                content = content.replace(pattern, replacement)
        
        with open(file_path, 'w') as f:
            f.write(content)
    
    async def _apply_maintainability_fix(
        self,
        file_path: str,
        change: Dict[str, Any]
    ):
        """Apply maintainability improvements"""
        # Use code generation agent to refactor
        with open(file_path, 'r') as f:
            original_content = f.read()
        
        refactored = await self.code_generation_agent.refactor_code(
            original_content,
            change["description"]
        )
        
        with open(file_path, 'w') as f:
            f.write(refactored)
    
    async def _generate_tests(
        self,
        file_path: str,
        change: Dict[str, Any]
    ):
        """Generate test file"""
        test_file_path = file_path.replace(".py", "_test.py")
        
        with open(file_path, 'r') as f:
            code_content = f.read()
        
        # Use code generation agent to create tests
        test_content = await self.code_generation_agent.generate_tests(code_content)
        
        with open(test_file_path, 'w') as f:
            f.write(test_content)
    
    async def _run_tests_sandboxed(
        self,
        sandbox_dir: str
    ) -> Dict[str, Any]:
        """Run tests in sandboxed environment"""
        try:
            # Run pytest in sandbox
            result = subprocess.run(
                ["pytest", sandbox_dir, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=sandbox_dir
            )
            
            return {
                "passed": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "output": "",
                "errors": "Test execution timeout"
            }
        except Exception as e:
            return {
                "passed": False,
                "output": "",
                "errors": str(e)
            }
    
    async def _request_hitl_approval(
        self,
        plan: RefactorPlan
    ) -> str:
        """Request HITL approval before commit"""
        approval_id = f"approval-{uuid.uuid4().hex[:12]}"
        
        print(f"HITL Approval Required for Refactoring Plan: {plan.plan_id}")
        print(f"Projects: {[p.name for p in plan.projects]}")
        print(f"Total changes: {len(plan.changes)}")
        print(f"Impact: {plan.estimated_impact}")
        print(f"Approval ID: {approval_id}")
        
        self.hitl_approvals[approval_id] = False
        
        return approval_id
    
    async def approve_changes(self, approval_id: str):
        """Approve changes for commit"""
        if approval_id in self.hitl_approvals:
            self.hitl_approvals[approval_id] = True
            
            # Find associated plan
            for plan in self.refactor_plans.values():
                if hasattr(plan, '_approval_id') and plan._approval_id == approval_id:
                    plan.approval_status = "approved"
                    break
    
    async def _commit_changes(
        self,
        project: CodeProject,
        modified_files: List[str]
    ) -> Dict[str, Any]:
        """Commit changes to repository"""
        sandbox_dir = self.sandbox_environments[project.project_id]
        
        try:
            repo = git.Repo(sandbox_dir)
            
            # Add modified files
            for file_path in modified_files:
                repo.index.add([file_path])
            
            # Commit
            commit_message = f"Automated refactoring - {len(modified_files)} files modified"
            commit = repo.index.commit(commit_message)
            
            return {
                "commit_hash": commit.hexsha,
                "message": commit_message,
                "files": modified_files
            }
        except Exception as e:
            return {
                "error": str(e)
            }


class FileSystemAgent:
    """
    Yi-Coder-9B based file system agent for code understanding
    """
    
    def __init__(self):
        self.model = "yi-coder-9b"
    
    async def scan_directory(
        self,
        directory: str
    ) -> Dict[str, CodeFile]:
        """Scan directory and analyze code files"""
        files = {}
        
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if self._is_code_file(filename):
                    file_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(file_path, directory)
                    
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    language = self._detect_language(filename)
                    
                    files[relative_path] = CodeFile(
                        path=relative_path,
                        content=content,
                        language=language,
                        original_content=content
                    )
        
        return files
    
    def _is_code_file(self, filename: str) -> bool:
        """Check if file is a code file"""
        code_extensions = [
            '.py', '.js', '.ts', '.java', '.go', '.rs',
            '.cpp', '.hpp', '.c', '.h', '.cs'
        ]
        return any(filename.endswith(ext) for ext in code_extensions)
    
    def _detect_language(self, filename: str) -> CodeLanguage:
        """Detect programming language from filename"""
        ext_to_lang = {
            '.py': CodeLanguage.PYTHON,
            '.js': CodeLanguage.JAVASCRIPT,
            '.ts': CodeLanguage.TYPESCRIPT,
            '.java': CodeLanguage.JAVA,
            '.go': CodeLanguage.GO,
            '.rs': CodeLanguage.RUST,
            '.cpp': CodeLanguage.CPP,
            '.cs': CodeLanguage.CSHARP
        }
        
        for ext, lang in ext_to_lang.items():
            if filename.endswith(ext):
                return lang
        
        return CodeLanguage.PYTHON  # Default


class CodeGenerationAgent:
    """
    DeepSeek-Coder-6.7B based code generation agent
    """
    
    def __init__(self):
        self.model = "deepseek-coder-6.7b"
    
    async def generate_code(
        self,
        language: CodeLanguage,
        requirements: str
    ) -> str:
        """Generate code based on requirements"""
        # In production, call DeepSeek-Coder model
        # For demo, generate sample code
        
        if language == CodeLanguage.PYTHON:
            return f'''"""
{requirements}
"""

def main():
    # Generated implementation
    pass

if __name__ == "__main__":
    main()
'''
        
        return f"// Generated code for {requirements}"
    
    async def optimize_code(
        self,
        code: str,
        optimization_goal: str
    ) -> str:
        """Optimize existing code"""
        # In production, use model to optimize
        # For demo, return slightly modified version
        
        optimized = code.replace("for i in range(len(", "for i, _ in enumerate(")
        optimized = optimized.replace("time.sleep", "await asyncio.sleep")
        
        return optimized
    
    async def refactor_code(
        self,
        code: str,
        refactor_description: str
    ) -> str:
        """Refactor code based on description"""
        # In production, use model to refactor
        # For demo, apply basic refactoring
        
        # Add type hints
        lines = code.split("\n")
        refactored_lines = []
        
        for line in lines:
            if line.strip().startswith("def ") and "->" not in line:
                # Add return type hint
                line = line.rstrip(":") + " -> None:"
            refactored_lines.append(line)
        
        return "\n".join(refactored_lines)
    
    async def generate_tests(
        self,
        code: str
    ) -> str:
        """Generate test cases for code"""
        # Extract function names
        functions = []
        for line in code.split("\n"):
            if line.strip().startswith("def "):
                func_name = line.split("(")[0].replace("def ", "").strip()
                if not func_name.startswith("_"):
                    functions.append(func_name)
        
        # Generate test template
        test_code = '''import pytest
import unittest
from unittest.mock import Mock, patch

class TestGenerated(unittest.TestCase):
    """Generated test cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        pass
    
    def tearDown(self):
        """Clean up after tests"""
        pass
'''
        
        for func in functions:
            test_code += f'''
    
    def test_{func}(self):
        """Test {func} function"""
        # TODO: Implement test for {func}
        pass
    
    def test_{func}_edge_cases(self):
        """Test {func} with edge cases"""
        # TODO: Add edge case tests
        pass
'''
        
        test_code += '''

if __name__ == "__main__":
    unittest.main()
'''
        
        return test_code


# Example usage
async def main():
    config = {}
    
    # Initialize orchestrator
    orchestrator = ApplicationBuildingOrchestrator(config)
    
    # Plan multi-repo refactor
    plan = await orchestrator.plan_multi_repo_refactor(
        repositories=[
            "https://github.com/example/service-a.git",
            "https://github.com/example/service-b.git"
        ],
        refactor_description="Improve performance and security across microservices",
        target_improvements={
            "performance": True,
            "security": True,
            "maintainability": True,
            "testing": True
        }
    )
    
    print(f"Refactor plan created: {plan.plan_id}")
    print(f"Projects: {[p.name for p in plan.projects]}")
    print(f"Total changes: {len(plan.changes)}")
    print(f"Estimated impact: {plan.estimated_impact}")
    
    # Approve if required
    if plan.requires_approval:
        approval_id = f"approval-{uuid.uuid4().hex[:12]}"
        await orchestrator.approve_changes(approval_id)
    
    # Execute refactor
    result = await orchestrator.execute_refactor_plan(plan.plan_id)
    print(f"Refactor result: {result}")


if __name__ == "__main__":
    asyncio.run(main())