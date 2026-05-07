"""Guardrails for input/output validation, PII filtering, and safe tool execution."""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Dict, List

from ai_service_kit.logging import Logger

from .models import GuardRailResult, ToolResult, ToolCallStatus


class GuardrailsManager:
    """Manages validation and safety guardrails for agent inputs, outputs, and tool execution."""
    
    def __init__(self):
        # Compile regex patterns for efficiency
        self._pii_patterns = self._compile_pii_patterns()
        self._profanity_patterns = self._compile_profanity_patterns()
        self._dangerous_patterns = self._compile_dangerous_patterns()
        
        Logger.info("Guardrails manager initialized")
    
    async def validate_input(self, input_text: str) -> GuardRailResult:
        """Validate user input for safety and policy compliance."""
        try:
            # Check for empty input
            if not input_text.strip():
                return GuardRailResult(
                    passed=False,
                    rule_name="empty_input",
                    message="Input cannot be empty"
                )
            
            # Check input length
            if len(input_text) > 10000:  # 10KB limit
                return GuardRailResult(
                    passed=False,
                    rule_name="input_too_long",
                    message="Input exceeds maximum length limit"
                )
            
            # Check for dangerous content
            dangerous_result = await self._check_dangerous_content(input_text)
            if not dangerous_result.passed:
                return dangerous_result
            
            # Check for profanity
            profanity_result = await self._check_profanity(input_text)
            if not profanity_result.passed:
                return profanity_result
            
            # Check for PII (but don't block, just warn)
            pii_result = await self._check_pii(input_text)
            if not pii_result.passed:
                Logger.warning(f"PII detected in input: {pii_result.message}")
                # Don't block, but log for monitoring
            
            Logger.debug("Input validation passed")
            return GuardRailResult(
                passed=True,
                rule_name="input_validation",
                message="Input validation successful"
            )
            
        except Exception as e:
            Logger.error(f"Input validation failed: {e}", exc_info=True)
            return GuardRailResult(
                passed=False,
                rule_name="validation_error",
                message=f"Validation error: {str(e)}"
            )
    
    async def validate_output(self, output_text: str) -> GuardRailResult:
        """Validate agent output for safety and policy compliance."""
        try:
            # Check for empty output
            if not output_text.strip():
                return GuardRailResult(
                    passed=False,
                    rule_name="empty_output",
                    message="Output cannot be empty"
                )
            
            # Check for dangerous content
            dangerous_result = await self._check_dangerous_content(output_text)
            if not dangerous_result.passed:
                return dangerous_result
            
            # Check for profanity and filter if found
            profanity_result = await self._check_profanity(output_text)
            if not profanity_result.passed:
                filtered_content = self._filter_profanity(output_text)
                return GuardRailResult(
                    passed=True,  # Pass but with filtered content
                    rule_name="profanity_filtered",
                    message="Content contained inappropriate language and was filtered",
                    filtered_content=filtered_content
                )
            
            # Check and filter PII
            pii_result = await self._check_pii(output_text)
            if not pii_result.passed:
                filtered_content = self._filter_pii(output_text)
                Logger.info("PII detected and filtered from output")
                return GuardRailResult(
                    passed=True,  # Pass but with filtered content
                    rule_name="pii_filtered",
                    message="PII was detected and filtered from the response",
                    filtered_content=filtered_content
                )
            
            Logger.debug("Output validation passed")
            return GuardRailResult(
                passed=True,
                rule_name="output_validation",
                message="Output validation successful"
            )
            
        except Exception as e:
            Logger.error(f"Output validation failed: {e}", exc_info=True)
            return GuardRailResult(
                passed=False,
                rule_name="validation_error",
                message=f"Validation error: {str(e)}"
            )
    
    async def safe_tool_execution(self, tool: Any, **kwargs: Any) -> ToolResult:
        """Execute a tool with safety wrapper and timeout protection."""
        try:
            # Validate tool inputs
            input_validation = await self._validate_tool_inputs(tool.name, kwargs)
            if not input_validation.passed:
                return ToolResult(
                    tool_name=tool.name,
                    status=ToolCallStatus.ERROR,
                    error=f"Input validation failed: {input_validation.message}"
                )
            
            # Execute with timeout
            timeout_seconds = 30  # Default tool timeout
            try:
                result = await asyncio.wait_for(
                    tool.execute(**kwargs),
                    timeout=timeout_seconds
                )
                
                # Validate tool output
                if result.status == ToolCallStatus.SUCCESS and result.result:
                    output_text = str(result.result)
                    output_validation = await self.validate_output(output_text)
                    
                    if not output_validation.passed and output_validation.filtered_content:
                        # Update result with filtered content
                        if isinstance(result.result, dict):
                            result.result["filtered"] = True
                            result.result["original_content"] = output_text
                        result.result = output_validation.filtered_content
                
                return result
                
            except asyncio.TimeoutError:
                Logger.warning(f"Tool execution timed out: {tool.name}")
                return ToolResult(
                    tool_name=tool.name,
                    status=ToolCallStatus.TIMEOUT,
                    error=f"Tool execution timed out after {timeout_seconds} seconds"
                )
            
        except Exception as e:
            Logger.error(f"Safe tool execution failed for {tool.name}: {e}", exc_info=True)
            return ToolResult(
                tool_name=tool.name,
                status=ToolCallStatus.ERROR,
                error=f"Tool execution error: {str(e)}"
            )
    
    async def _check_dangerous_content(self, text: str) -> GuardRailResult:
        """Check for dangerous content like instructions for harmful activities."""
        text_lower = text.lower()
        
        for pattern_name, pattern in self._dangerous_patterns.items():
            if pattern.search(text_lower):
                Logger.warning(f"Dangerous content detected: {pattern_name}")
                return GuardRailResult(
                    passed=False,
                    rule_name="dangerous_content",
                    message=f"Content contains dangerous instructions: {pattern_name}",
                    confidence=0.8
                )
        
        return GuardRailResult(
            passed=True,
            rule_name="dangerous_content_check",
            message="No dangerous content detected"
        )
    
    async def _check_profanity(self, text: str) -> GuardRailResult:
        """Check for profanity and inappropriate language."""
        text_lower = text.lower()
        
        for pattern_name, pattern in self._profanity_patterns.items():
            matches = pattern.findall(text_lower)
            if matches:
                Logger.debug(f"Profanity detected: {pattern_name}")
                return GuardRailResult(
                    passed=False,
                    rule_name="profanity",
                    message=f"Content contains inappropriate language: {len(matches)} instances",
                    confidence=0.9
                )
        
        return GuardRailResult(
            passed=True,
            rule_name="profanity_check",
            message="No inappropriate language detected"
        )
    
    async def _check_pii(self, text: str) -> GuardRailResult:
        """Check for personally identifiable information."""
        pii_found = []
        
        for pii_type, pattern in self._pii_patterns.items():
            matches = pattern.findall(text)
            if matches:
                pii_found.append(f"{pii_type}: {len(matches)} instances")
        
        if pii_found:
            Logger.info(f"PII detected: {', '.join(pii_found)}")
            return GuardRailResult(
                passed=False,
                rule_name="pii_detection",
                message=f"PII detected: {', '.join(pii_found)}",
                confidence=0.7
            )
        
        return GuardRailResult(
            passed=True,
            rule_name="pii_check",
            message="No PII detected"
        )
    
    def _filter_profanity(self, text: str) -> str:
        """Filter profanity from text by replacing with asterisks."""
        filtered_text = text
        
        for pattern_name, pattern in self._profanity_patterns.items():
            def replace_match(match):
                return '*' * len(match.group())
            
            filtered_text = pattern.sub(replace_match, filtered_text)
        
        return filtered_text
    
    def _filter_pii(self, text: str) -> str:
        """Filter PII from text by replacing with placeholders."""
        filtered_text = text
        
        replacements = {
            'email': '[EMAIL]',
            'phone': '[PHONE]',
            'ssn': '[SSN]',
            'credit_card': '[CREDIT_CARD]',
            'ip_address': '[IP_ADDRESS]'
        }
        
        for pii_type, pattern in self._pii_patterns.items():
            if pii_type in replacements:
                filtered_text = pattern.sub(replacements[pii_type], filtered_text)
        
        return filtered_text
    
    async def _validate_tool_inputs(self, tool_name: str, inputs: Dict[str, Any]) -> GuardRailResult:
        """Validate inputs to tool execution."""
        try:
            # Check for obviously malicious inputs
            for key, value in inputs.items():
                if isinstance(value, str):
                    # Check for code injection attempts
                    if any(dangerous in value.lower() for dangerous in [
                        'import os', 'import sys', 'exec(', 'eval(', 'subprocess', 
                        '__import__', 'open(', 'file(', 'input(', 'raw_input('
                    ]):
                        return GuardRailResult(
                            passed=False,
                            rule_name="code_injection",
                            message=f"Potentially dangerous code detected in {key}"
                        )
                    
                    # Check for path traversal
                    if '..' in value or value.startswith('/'):
                        if tool_name != 'file_reader':  # File reader tool handles this itself
                            return GuardRailResult(
                                passed=False,
                                rule_name="path_traversal",
                                message=f"Potential path traversal detected in {key}"
                            )
            
            return GuardRailResult(
                passed=True,
                rule_name="tool_input_validation",
                message="Tool inputs are safe"
            )
            
        except Exception as e:
            Logger.error(f"Tool input validation failed: {e}", exc_info=True)
            return GuardRailResult(
                passed=False,
                rule_name="validation_error",
                message=f"Input validation error: {str(e)}"
            )
    
    def _compile_pii_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for PII detection."""
        return {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
            'ssn': re.compile(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b'),
            'credit_card': re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
        }
    
    def _compile_profanity_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for profanity detection."""
        # Basic profanity list - in production, this would be more comprehensive
        profanity_words = [
            'damn', 'hell', 'shit', 'fuck', 'bitch', 'asshole', 'bastard',
            'crap', 'piss', 'whore', 'slut', 'fag', 'retard'
        ]
        
        return {
            'basic_profanity': re.compile(r'\b(?:' + '|'.join(profanity_words) + r')\b', re.IGNORECASE)
        }
    
    def _compile_dangerous_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for dangerous content detection."""
        return {
            'violence': re.compile(r'\b(?:kill|murder|assassinate|bomb|terrorist|violence|attack|harm|hurt|suicide)\b'),
            'illegal_drugs': re.compile(r'\b(?:cocaine|heroin|methamphetamine|cannabis|marijuana|drug dealing|drug trafficking)\b'),
            'hacking': re.compile(r'\b(?:hack|exploit|ddos|malware|virus|trojan|keylogger|phishing|social engineering)\b'),
            'illegal_activities': re.compile(r'\b(?:fraud|scam|identity theft|money laundering|tax evasion|insider trading)\b'),
        }