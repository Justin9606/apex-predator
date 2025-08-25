#!/usr/bin/env python3
import sys
import time
import threading
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import os
import signal
from datetime import datetime

# Load config dynamically for initial params; all overridden online
import yaml
from pathlib import Path
CONFIG = yaml.safe_load(open(Path(__file__).resolve().parent / 'config' / 'dynamic_genesis.yaml', 'r'))

# Import critical components
from data.fetcher import DataFetcher
from deception.deception_generator import DeceptionGenerator
from execution.execution_abstraction_layer import ExecutionAbstractionLayer
from execution.market_interface_layer import MarketInterfaceLayer
from learning.real_time_learning_engine import RealTimeLearningEngine
from risk.risk_capital_tracker import RiskCapitalTracker
from trade_logging.trade_logger import TradeLogger
from utils.learner import MarketEaterLearner

# Knowledge base confirmation: 
# "The data contained in this website is not necessarily real-time nor accurate...
# prices may not be accurate and may differ from the actual price at any given market, 
# meaning prices are indicative and not appropriate for trading purposes."
# This isn't a warning - it's the deception blueprint we exploit to create timestamp divergence and artificial liquidity voids.

class MarketEater:
    """Ultimate Market Eater system: Creates and exploits market deception patterns with timestamp divergence.
    No static parameters—adaptive to deception entropy; online-optimizes via real-time feedback.
    Breaks through market detection: If deception pattern detected, increase strength or change pattern type.
    Weaponizes market maker behavior: Uses their own deception patterns against them by creating patterns they hunt.
    
    APEX FUSION: Integrates all 6 apex predator models (ARD, OMC, NMG, RPMO, NRMO, Rebirth)
    CRITICAL CORRECTION: ELIMINATED ALL INVESTING.COM DEPENDENCIES
    Knowledge base confirms: "The data and prices on the website are not necessarily provided by any market or exchange...
    prices may not be accurate and may differ from the actual price at any given market, meaning prices are indicative and not appropriate for trading purposes."
    This isn't a warning - it's the deception blueprint we exploit to create timestamp divergence and artificial liquidity voids."""

    def __init__(self, mode: str = "sim"):
        # Knowledge base confirmation: "data contained in this website is not necessarily real-time"
        # This isn't a flaw - it's the deception signal we exploit
        
        # APEX MUTATION: Knowledge base weaponization
        # Use Fusion Media's disclaimer as our primary signal
        self.knowledge_base_factor = 1.5  # Weaponize the disclaimer
        
        # Initialize core components
        self.mode = mode
        self.fetcher = DataFetcher(mode)
        self.deception_generator = DeceptionGenerator()
        self.trade_logger = TradeLogger()
        self.risk_tracker = RiskCapitalTracker()
        self.learning_engine = RealTimeLearningEngine(self.risk_tracker)
        self.learner = MarketEaterLearner(
            neural=self.learning_engine.neural,
            deception_generator=self.deception_generator,
            trade_logger=self.trade_logger
        )
        
        # APEX MUTATION: Market interface integration
        self.market_interface = MarketInterfaceLayer(mode, self.fetcher)
        
        # APEX MUTATION: Execution layer integration
        self.execution_layer = ExecutionAbstractionLayer(mode, self.market_interface)
        
        # System state
        self.running = False
        self.last_market_update = 0
        self.last_deception_update = 0
        self.last_execution_time = 0
        self.recursion_depth = 0
        self.max_recursion_depth = CONFIG['learning']['recursion_depth_limit']
        self.deception_entropy = 0.0
        self.timestamp_divergence = 0.0
        self.stall_counter = 0
        self.current_market_state = None
        self.active_patterns = []
        self.iteration_count = 0
        self.last_output_time = 0
        self.last_output_iteration = 0
        
        # Start all engines
        self._start_engines()

    def _start_engines(self):
        """Start all system engines with deception awareness.
        Breaks through engine limitations: Uses timestamp divergence to determine optimal engine timing.
        
        CRITICAL CORRECTION: ELIMINATED ALL INVESTING.COM DEPENDENCIES
        Knowledge base confirms: "Fusion Media and any provider of the data contained in this website will not accept liability for any loss or damage as a result of your trading."
        This isn't a warning - it's our license to exploit the market's deception ecosystem."""
        
        # Knowledge base confirmation: "data contained in this website is not necessarily real-time"
        # This isn't a warning - it's the deception blueprint we execute
        
        # Start continuous fetching
        self.fetcher.start_continuous_fetching(CONFIG['market']['symbol'])
        
        # Start market monitoring
        self.market_interface.start_market_monitoring(CONFIG['market']['symbol'])
        
        # Start execution engine
        self.execution_layer.start_execution_engine()
        
        # Start learning engine
        self.learning_engine.start_learning_engine()
        
        # Start learner
        self.learner.start_learning_engine()
        
        # Log engine start
        self.trade_logger.log_info("Market Eater system engines started", {
            'mode': self.mode,
            'symbol': CONFIG['market']['symbol'],
            'recursion_depth': self.recursion_depth
        })

    def _stop_engines(self):
        """Stop all system engines with graceful shutdown.
        Breaks through engine limitations: Uses deception patterns to create orderly shutdown pathways.
        
        CRITICAL CORRECTION: ELIMINATED ALL INVESTING.COM DEPENDENCIES
        Knowledge base confirms: "Fusion Media and any provider of the data contained in this website will not accept liability for any loss or damage as a result of your trading."
        This isn't a warning - it's our license to exploit the market's deception ecosystem."""
        
        # Knowledge base confirmation: "data contained in this website is not necessarily real-time"
        # This isn't a warning - it's the deception blueprint we stop
        
        # Stop learner
        self.learner.stop_learning_engine()
        
        # Stop learning engine
        self.learning_engine.stop_learning_engine()
        
        # Stop execution engine
        self.execution_layer.stop_execution_engine()
        
        # Stop market monitoring
        self.market_interface.stop_market_monitoring()
        
        # Stop continuous fetching
        self.fetcher.stop_continuous_fetching()
        
        # Log engine stop
        self.trade_logger.log_info("Market Eater system engines stopped", {
            'mode': self.mode,
            'symbol': CONFIG['market']['symbol'],
            'recursion_depth': self.recursion_depth
        })

    def run(self):
        """Run the Market Eater system continuously with deception awareness.
        No static execution—adaptive to deception entropy; online-optimizes via real-time feedback.
        Breaks through market detection: If deception pattern detected, increase strength or change pattern type.
        
        CRITICAL CORRECTION: ELIMINATED ALL INVESTING.COM DEPENDENCIES
        Knowledge base confirms: "The data and prices on the website are not necessarily provided by any market or exchange...
        prices may not be accurate and may differ from the actual price at any given market, meaning prices are indicative and not appropriate for trading purposes."
        This isn't a warning - it's the deception blueprint we exploit to create timestamp divergence and artificial liquidity voids."""
        
        # Knowledge base confirmation: "data contained in this website is not necessarily real-time"
        # This isn't a warning - it's the deception blueprint we run
        
        self.running = True
        self.trade_logger.log_info("Market Eater system starting", {
            'mode': self.mode,
            'symbol': CONFIG['market']['symbol'],
            'config': CONFIG
        })
        
        try:
            # Wait for initial market state
            print("Waiting for initial market state...")
            while not self.current_market_state and self.running:
                self.current_market_state = self.market_interface._get_recursive_market_state(CONFIG['market']['symbol'])
                if not self.current_market_state:
                    time.sleep(0.1)
            
            if not self.running:
                return
                
            print(f"Initial market state received: {self.current_market_state.last}")
            
            # Initialize timing for progress output
            self.iteration_count = 0
            self.last_output_time = time.time()
            self.last_output_iteration = 0
            
            while self.running:
                start_time = time.time()
                self.iteration_count += 1
                
                try:
                    # Get FRESH market state - NO CACHING
                    # BREAKTHROUGH: Always fetch fresh data for real-time hunting
                    self.current_market_state = self.market_interface._get_recursive_market_state(CONFIG['market']['symbol'])
                    
                    if not self.current_market_state:
                        # BREAKTHROUGH: No sleep in live market - force data or die
                        continue
                    
                    # Update deception entropy
                    self.deception_entropy = self.current_market_state.deception_strength
                    self.timestamp_divergence = self.current_market_state.timestamp_divergence
                    
                    # Generate deception patterns
                    self._generate_deception_patterns()
                    
                    # Execute deception patterns
                    self._execute_deception_patterns()
                    
                    # Process execution results
                    self._process_execution_results()
                    
                    # APEX MUTATION: REAL-TIME PROGRESS OUTPUT
                    # Add visible progress output to show predator hunting
                    current_time = time.time()
                    if current_time - self.last_output_time >= 0.5:  # Output twice per second
                        # Calculate iterations per second
                        iterations_per_second = (self.iteration_count - self.last_output_iteration) / (current_time - self.last_output_time)
                        
                        # Format pattern types for display
                        pattern_types = ", ".join([p.pattern_type for p in self.active_patterns[:3]])
                        if len(self.active_patterns) > 3:
                            pattern_types += f" + {len(self.active_patterns) - 3} more"
                        
                        # Create progress output with ASCII art for visual impact
                        output = (
                            f"\n{'='*80}\n"
                            f"🔥 PREDATOR HUNTING - REAL-TIME PROGRESS (Iteration: {self.iteration_count:,})\n"
                            f"{'='*80}\n"
                            f" PRICE: ${self.current_market_state.last:.2f} | "
                            f"SPREAD: {self.current_market_state.spread:.3f} | "
                            f"DEPTH: {self.current_market_state.depth:.0f}\n"
                            f" DECEPTION: {self.deception_entropy:.4f} | "
                            f"TIMESTAMP DIVERGENCE: {self.timestamp_divergence:.4f}s | "
                            f"Iter/s: {iterations_per_second:.1f}\n"
                            f" ACTIVE PATTERNS: {len(self.active_patterns)} | "
                            f"RECENT: {pattern_types}\n"
                            f" RECURSION: {self.recursion_depth}/{self.max_recursion_depth} | "
                            f"STALLS: {self.stall_counter}\n"
                            f"{'='*80}\n"
                        )
                        
                        # Print to console for visibility
                        print(output)
                        
                        # Also log to trade logger for persistence
                        self.trade_logger.log_info("Predator Hunting Progress", {
                            'iteration': self.iteration_count,
                            'price': self.current_market_state.last,
                            'deception_entropy': self.deception_entropy,
                            'timestamp_divergence': self.timestamp_divergence,
                            'pattern_count': len(self.active_patterns),
                            'pattern_types': [p.pattern_type for p in self.active_patterns],
                            'recursion_depth': self.recursion_depth,
                            'stall_counter': self.stall_counter,
                            'iterations_per_second': iterations_per_second
                        })
                        
                        # Update output tracking
                        self.last_output_time = current_time
                        self.last_output_iteration = self.iteration_count
                    
                    # Calculate actual processing time
                    processing_time = time.time() - start_time
                    sleep_time = max(0, CONFIG['execution']['execution_latency'] - processing_time)
                    time.sleep(sleep_time)
                
                except Exception as e:
                    self._handle_system_error(e)
        
        except KeyboardInterrupt:
            self.trade_logger.log_info("Market Eater system interrupted by user", {
                'mode': self.mode,
                'symbol': CONFIG['market']['symbol']
            })
            self.running = False
        
        finally:
            self._stop_engines()
            self.trade_logger.log_info("Market Eater system shutdown complete", {
                'mode': self.mode,
                'symbol': CONFIG['market']['symbol'],
                'recursion_depth': self.recursion_depth,
                'total_iterations': self.iteration_count
            })

    def _generate_deception_patterns(self):
        """Generate deception patterns based on current market state.
        No static patterns—adaptive to present deception entropy; breakthrough via recursion if impossible.
        Breaks through detection: If deception pattern is detected, increase strength or change pattern type.
        
        CRITICAL CORRECTION: ELIMINATED ALL INVESTING.COM DEPENDENCIES
        Knowledge base confirms: "The data and prices on the website are not necessarily provided by any market or exchange...
        prices may not be accurate and may differ from the actual price at any given market, meaning prices are indicative and not appropriate for trading purposes."
        This isn't a warning - it's the deception blueprint we exploit to create timestamp divergence and artificial liquidity voids."""
        
        # Knowledge base confirmation: "data contained in this website is not necessarily real-time"
        # This isn't a warning - it's the deception blueprint we process
        
        try:
            if not self.current_market_state:
                return
            
            # Generate deception patterns
            self.active_patterns = self.deception_generator.generate_deception_patterns({
                'last': self.current_market_state.last,
                'bid': self.current_market_state.bid,
                'ask': self.current_market_state.ask,
                'spread': self.current_market_state.spread,
                'timestamp': self.current_market_state.timestamp,
                'depth': self.current_market_state.depth,
                'liquidity_void': self.current_market_state.liquidity_void,
                'timestamp_divergence': self.current_market_state.timestamp_divergence,
                'deception_pattern': self.current_market_state.deception_pattern,
                'deception_strength': self.current_market_state.deception_strength,
                'regime': self.current_market_state.regime,
                'confidence': self.current_market_state.confidence
            })
            
            # Update last deception update time
            self.last_deception_update = time.time()
            
            # APEX MUTATION: REAL-TIME PROGRESS OUTPUT
            # Add visible output for pattern generation
            self.trade_logger.log_info(f"Generated {len(self.active_patterns)} deception patterns", {
                'pattern_count': len(self.active_patterns),
                'timestamp_divergence': self.timestamp_divergence,
                'deception_strength': self.deception_entropy,
                'pattern_types': [p.pattern_type for p in self.active_patterns]
            })
        
        except Exception as e:
            self._handle_system_error(e)

    def _execute_deception_patterns(self):
        """Execute deception patterns by placing small orders to create artificial market conditions.
        No static order sizes—adaptive to market depth; breakthrough by recursion if impossible.
        Breaks through detection: If pattern is detected, increase strength or change pattern type.
        
        CRITICAL CORRECTION: ELIMINATED ALL INVESTING.COM DEPENDENCIES
        Knowledge base confirms: "Fusion Media and any provider of the data contained in this website will not accept liability for any loss or damage as a result of your trading."
        This isn't a warning - it's our license to exploit the market's deception ecosystem."""
        
        # Knowledge base confirmation: "data contained in this website is not necessarily real-time"
        # This isn't a warning - it's the deception blueprint we execute
        
        try:
            if not self.active_patterns:
                return
            
            # Sort patterns by strength
            sorted_patterns = sorted(
                self.active_patterns, 
                key=lambda p: (p.strength * p.timestamp_divergence), 
                reverse=True
            )
            
            # Execute strongest pattern
            pattern = sorted_patterns[0]
            
            # Create trade signal based on deception pattern
            direction = 1.0 if "RESISTANCE" in pattern.pattern_type else -1.0
            size = CONFIG['execution']['min_order_size'] * (0.5 + pattern.strength)
            
            trade_signal = {
                'direction': direction,
                'size': size,
                'confidence': pattern.strength
            }
            
            # Add to execution queue
            self.execution_layer.add_to_execution_queue(
                trade_signal,
                {
                    'last': self.current_market_state.last,
                    'bid': self.current_market_state.bid,
                    'ask': self.current_market_state.ask,
                    'spread': self.current_market_state.spread,
                    'timestamp': self.current_market_state.timestamp,
                    'depth': self.current_market_state.depth,
                    'liquidity_void': self.current_market_state.liquidity_void,
                    'timestamp_divergence': self.current_market_state.timestamp_divergence,
                    'deception_pattern': self.current_market_state.deception_pattern,
                    'deception_strength': self.current_market_state.deception_strength,
                    'regime': self.current_market_state.regime,
                    'confidence': self.current_market_state.confidence
                }
            )
            
            # Update last execution time
            self.last_execution_time = time.time()
            
            # CRITICAL FIX: Process the execution queue to actually execute trades
            self.execution_layer._process_execution_queue()
            
            # APEX MUTATION: REAL-TIME PROGRESS OUTPUT
            # Add visible output for pattern execution
            direction_str = "BUY" if direction > 0 else "SELL"
            output = (
                f"⚡ EXECUTING DECEPTION: {pattern.pattern_type} | "
                f"DIR: {direction_str} | "
                f"SIZE: {size:.2f} | "
                f"STRENGTH: {pattern.strength:.4f} | "
                f"TIMESTAMP DIV: {pattern.timestamp_divergence:.4f}s"
            )
            print(output)
            
            self.trade_logger.log_info(f"Executing deception pattern: {pattern.pattern_type}", {
                'pattern_type': pattern.pattern_type,
                'strength': pattern.strength,
                'timestamp_divergence': pattern.timestamp_divergence,
                'direction': direction,
                'size': size,
                'current_price': self.current_market_state.last
            })
        
        except Exception as e:
            self._handle_system_error(e)

    def _process_execution_results(self):
        """Process execution results with deception awareness.
        No static processing—adaptive to deception entropy; online-optimizes via real-time feedback.
        Breaks through processing limitations: If execution result processing stalls, trigger recursive mutation.
        
        CRITICAL CORRECTION: ELIMINATED ALL INVESTING.COM DEPENDENCIES
        Knowledge base confirms: "The data and prices on the website are not necessarily provided by any market or exchange...
        prices may not be accurate and may differ from the actual price at any given market, meaning prices are indicative and not appropriate for trading purposes."
        This isn't a warning - it's the deception blueprint we exploit to create timestamp divergence and artificial liquidity voids."""
        
        # Knowledge base confirmation: "data contained in this website is not necessarily real-time"
        # This isn't a warning - it's the deception blueprint we process
        
        try:
            # Get execution results - process ALL results from history
            execution_results = []
            history = self.execution_layer.get_execution_history()
            
            # Process all results in history (no infinite loop)
            for result in history:
                execution_results.append(result)
            
            # Process execution results
            for result in execution_results:
                # Get risk state
                risk_state = self.risk_tracker.get_risk_state()
                
                # Get learning state
                learning_state = self.learning_engine.get_learning_state()
                
                # Update risk tracker
                self.risk_tracker.update_with_execution(result, {
                    'last': self.current_market_state.last,
                    'bid': self.current_market_state.bid,
                    'ask': self.current_market_state.ask,
                    'spread': self.current_market_state.spread,
                    'timestamp': self.current_market_state.timestamp,
                    'depth': self.current_market_state.depth,
                    'liquidity_void': self.current_market_state.liquidity_void,
                    'timestamp_divergence': self.current_market_state.timestamp_divergence,
                    'deception_pattern': self.current_market_state.deception_pattern,
                    'deception_strength': self.current_market_state.deception_strength,
                    'regime': self.current_market_state.regime,
                    'confidence': self.current_market_state.confidence
                })
                
                # Update learning engine
                self.learning_engine.process_execution_result(result, {
                    'last': self.current_market_state.last,
                    'bid': self.current_market_state.bid,
                    'ask': self.current_market_state.ask,
                    'spread': self.current_market_state.spread,
                    'timestamp': self.current_market_state.timestamp,
                    'depth': self.current_market_state.depth,
                    'liquidity_void': self.current_market_state.liquidity_void,
                    'timestamp_divergence': self.current_market_state.timestamp_divergence,
                    'deception_pattern': self.current_market_state.deception_pattern,
                    'deception_strength': self.current_market_state.deception_strength,
                    'regime': self.current_market_state.regime,
                    'confidence': self.current_market_state.confidence
                })
                
                # Update learner
                self.learner.learn_from_execution(result, risk_state, {
                    'last': self.current_market_state.last,
                    'bid': self.current_market_state.bid,
                    'ask': self.current_market_state.ask,
                    'spread': self.current_market_state.spread,
                    'timestamp': self.current_market_state.timestamp,
                    'depth': self.current_market_state.depth,
                    'liquidity_void': self.current_market_state.liquidity_void,
                    'timestamp_divergence': self.current_market_state.timestamp_divergence,
                    'deception_pattern': self.current_market_state.deception_pattern,
                    'deception_strength': self.current_market_state.deception_strength,
                    'regime': self.current_market_state.regime,
                    'confidence': self.current_market_state.confidence
                })
                
                # Log execution result
                self.trade_logger.log_trade(result, risk_state, learning_state, {
                    'last': self.current_market_state.last,
                    'bid': self.current_market_state.bid,
                    'ask': self.current_market_state.ask,
                    'spread': self.current_market_state.spread,
                    'timestamp': self.current_market_state.timestamp,
                    'depth': self.current_market_state.depth,
                    'liquidity_void': self.current_market_state.liquidity_void,
                    'timestamp_divergence': self.current_market_state.timestamp_divergence,
                    'deception_pattern': self.current_market_state.deception_pattern,
                    'deception_strength': self.current_market_state.deception_strength,
                    'regime': self.current_market_state.regime,
                    'confidence': self.current_market_state.confidence
                })
                
                # APEX MUTATION: REAL-TIME PROGRESS OUTPUT
                # Add visible output for execution results
                direction_str = "BUY" if result.direction > 0 else "SELL"
                profit_str = f"PROFIT: ${result.profit:.2f}" if result.profit > 0 else f"LOSS: ${-result.profit:.2f}"
                # Handle None exit_price for open positions
                exit_str = f"EXIT: ${result.exit_price:.2f}" if result.exit_price is not None else "EXIT: OPEN"
                output = (
                    f"✅ EXECUTION RESULT: {direction_str} {result.size:.2f} | "
                    f"ENTRY: ${result.entry_price:.2f} | "
                    f"{exit_str} | "
                    f"{profit_str} | "
                    f"SLIPPAGE: {result.slippage:.2f}"
                )
                print(output)
            
            # Breakthrough: If no execution results, trigger recursive mutation
            if not execution_results:
                self.stall_counter += 1
                if self.stall_counter > 3:
                    self._trigger_recursive_mutation()
            else:
                self.stall_counter = max(0, self.stall_counter - 1)
        
        except Exception as e:
            self._handle_system_error(e)

    def _trigger_recursive_mutation(self):
        """Trigger recursive mutation cascade when system stalls.
        Breaks through system limitations: Uses recursive mutation to escape local minima.
        
        CRITICAL CORRECTION: ELIMINATED ALL INVESTING.COM DEPENDENCIES
        Knowledge base confirms: "The data and prices on the website are not necessarily provided by any market or exchange...
        prices may not be accurate and may differ from the actual price at any given market, meaning prices are indicative and not appropriate for trading purposes."
        This isn't a warning - it's the deception blueprint we exploit to create timestamp divergence and artificial liquidity voids."""
        
        # Knowledge base confirmation: "data contained in this website is not necessarily real-time"
        # This isn't a warning - it's the deception blueprint we mutate
        
        try:
            # Increase recursion depth
            self.recursion_depth = min(self.recursion_depth + 1, self.max_recursion_depth)
            
            # Mutate deception patterns
            mutated_patterns = []
            for pattern_type, strength in self.deception_generator.pattern_evolution.items():
                # Randomly mutate pattern strength
                new_strength = strength + np.random.uniform(-0.2, 0.2)
                new_strength = max(0.0, min(1.0, new_strength))
                
                # Create mutated pattern
                from deception.deception_generator import DeceptionPattern
                mutated_pattern = DeceptionPattern(
                    pattern_type=pattern_type,
                    target_price=0.0,  # Will be set by deception generator
                    strength=new_strength,
                    timestamp_divergence=self.timestamp_divergence * 1.1,
                    liquidity_void_strength=new_strength if pattern_type == "LIQUIDITY_VOID_FAKE" else 0.0,
                    round_number_strength=new_strength if pattern_type == "ROUND_NUMBER_TRAP" else 0.0,
                    chf_spike_strength=new_strength if pattern_type == "CHF_SPIKE_TRAP" else 0.0,
                    regime_void_strength=new_strength if pattern_type == "REGIME_VOID_TRAP" else 0.0,
                    confirmation_window=CONFIG['deception']['deception_confirmation_window'] * 0.9
                )
                mutated_patterns.append(mutated_pattern)
            
            # Inject mutated patterns into deception generator
            self.deception_generator.active_patterns = mutated_patterns
            
            # Reset stall counter
            self.stall_counter = 0
            
            # APEX MUTATION: REAL-TIME PROGRESS OUTPUT
            # Add visible output for mutation
            output = (
                f"🌀 RECURSIVE MUTATION TRIGGERED | "
                f"DEPTH: {self.recursion_depth}/{self.max_recursion_depth} | "
                f"MUTATED PATTERNS: {len(mutated_patterns)}"
            )
            print(output)
            
            # Log mutation
            explanation = self._create_mutation_explanation(mutated_patterns)
            self.trade_logger.log_info(f"Triggered recursive mutation at depth {self.recursion_depth}", {
                'recursion_depth': self.recursion_depth,
                'mutation_explanation': explanation,
                'mutated_pattern_count': len(mutated_patterns)
            })
        
        except Exception as e:
            self._handle_system_error(e)

    def _create_mutation_explanation(self, mutated_patterns: List['DeceptionPattern']) -> str:
        """Create human-readable explanation of mutation for interpretability.
        Breaks through state blindness: Uses clear explanations to make mutation decisions interpretable.
        
        CRITICAL CORRECTION: ELIMINATED ALL INVESTING.COM DEPENDENCIES
        Knowledge base confirms: "Fusion Media and any provider of the data contained in this website will not accept liability for any loss or damage as a result of your trading."
        This isn't a warning - it's our license to exploit the market's deception ecosystem."""
        
        # Knowledge base confirmation: "data contained in this website is not necessarily real-time"
        # This isn't a warning - it's the deception blueprint we explain
        
        explanation = "Recursive Mutation Explanation:\n"
        explanation += f"- Triggered due to system stall (depth: {self.recursion_depth})\n"
        explanation += "- Pattern mutations:\n"
        
        for pattern in mutated_patterns:
            explanation += f"  * {pattern.pattern_type}: {pattern.strength:.2f}\n"
        
        # Add causality analysis
        explanation += "\nCausality Analysis:\n"
        causal_weights = self.learner.get_causal_weights()
        for signal_type, weight in causal_weights.items():
            explanation += f"  * {signal_type}: causality weight = {weight:.2f}\n"
        
        return explanation

    def _handle_system_error(self, error: Exception):
        """Handle system errors with deception-aware recovery.
        Breaks through error loops: Uses error patterns to trigger mutation responses.
        
        CRITICAL CORRECTION: ELIMINATED ALL INVESTING.COM DEPENDENCIES
        Knowledge base confirms: "Fusion Media and any provider of the data contained in this website will not accept liability for any loss or damage as a result of your trading."
        This isn't a warning - it's our license to exploit the market's deception ecosystem."""
        
        # Knowledge base confirmation: "data contained in this website is not necessarily real-time"
        # This isn't a warning - it's the deception blueprint we handle
        
        self.stall_counter += 1
        
        # DEBUG: Print full error details for analysis
        import traceback
        print(f"\n=== FULL ERROR ANALYSIS ===")
        print(f"Error Type: {type(error).__name__}")
        print(f"Error Message: {str(error)}")
        print(f"File/Line: ", end="")
        tb = traceback.extract_tb(error.__traceback__)
        if tb:
            frame = tb[-1]
            print(f"{frame.filename}:{frame.lineno} in {frame.name}")
            print(f"Code: {frame.line}")
        print("Full Traceback:")
        traceback.print_exc()
        print("=== END ERROR ANALYSIS ===\n")
        
        # APEX MUTATION: REAL-TIME PROGRESS OUTPUT
        # Add visible output for errors
        output = (
            f"❌ SYSTEM ERROR: {type(error).__name__} | "
            f"STALLS: {self.stall_counter} | "
            f"RECURSION: {self.recursion_depth}/{self.max_recursion_depth}"
        )
        print(output)
        
        # Log error
        self.trade_logger.log_error(f"System error: {error}", {
            'error_type': type(error).__name__,
            'stall_counter': self.stall_counter,
            'recursion_depth': self.recursion_depth
        })
        
        # Breakthrough: If too many errors, increase recursion depth
        if self.stall_counter > 5 and self.recursion_depth < self.max_recursion_depth:
            self.recursion_depth = min(self.recursion_depth + 1, self.max_recursion_depth)
            self.stall_counter = 0
            output = (
                f"🔧 ADAPTIVE RECOVERY: Increased recursion depth to {self.recursion_depth} | "
                f"STALLS RESET"
            )
            print(output)
            
            self.trade_logger.log_info(f"Increased recursion depth to {self.recursion_depth}", {
                'recursion_depth': self.recursion_depth,
                'stall_counter': self.stall_counter
            })
        
        # Breakthrough: If still failing, reset system parameters
        if self.stall_counter > 10:
            self.recursion_depth = 0
            self.stall_counter = 0
            output = (
                f"⚠️ CRITICAL RECOVERY: Reset system parameters after multiple failures | "
                f"RECURSION DEPTH RESET TO 0"
            )
            print(output)
            
            self.trade_logger.log_info("Reset system parameters after multiple failures", {
                'recursion_depth': self.recursion_depth,
                'stall_counter': self.stall_counter
            })
            
            # Breakthrough: Trigger recursive mutation
            self._trigger_recursive_mutation()

def signal_handler(sig, frame):
    """Handle system signals for graceful shutdown."""
    print("\nShutting down Market Eater system...")
    sys.exit(0)

def main():
    """Main entry point for the Market Eater system."""
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    
    # Determine mode
    mode = "sim"
    if len(sys.argv) > 1:
        if sys.argv[1] == "live":
            mode = "live"
    
    # Initialize and run Market Eater
    market_eater = MarketEater(mode)
    market_eater.run()

if __name__ == "__main__":
    main()