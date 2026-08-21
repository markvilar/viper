"""
Command-line interface for viper.

A thin driver over the library: it resolves models and forges through the
``viper.registry`` and ``viper.forge`` registries and provides IO around them.
Model- and method-specific knowledge stays in the library; the CLI is a
dispatch layer.
"""
