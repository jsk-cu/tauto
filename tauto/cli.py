"""
Command line interface for TAuto.
"""

import click
from pathlib import Path
import yaml
import sys
import os
from typing import Optional

from tauto.config import load_config, save_config, get_default_config
from tauto.utils import get_logger, configure_logging, setup_wandb, log_config


@click.group()
@click.version_option()
def cli():
    """TAuto: PyTorch Model Optimization Framework CLI."""
    pass


@cli.command()
@click.option("--output", "-o", type=click.Path(), default="tauto_config.yaml", 
              help="Output file path for the configuration")
@click.option("--force", "-f", is_flag=True, help="Force overwrite if the file exists")
def init_config(output: str, force: bool):
    """Initialize a default configuration file."""
    output_path = Path(output)
    
    if output_path.exists() and not force:
        click.echo(f"Error: {output_path} already exists. Use --force to overwrite.")
        sys.exit(1)
    
    # Get default configuration
    config = get_default_config()
    
    # Save configuration
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    click.echo(f"Configuration saved to {output_path}")


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), required=True,
              help="Path to the configuration file")
def validate_config(config: str):
    """Validate a configuration file."""
    try:
        config_obj = load_config(config)
        click.echo(f"Configuration file {config} is valid")
    except Exception as e:
        click.echo(f"Error validating configuration: {e}")
        sys.exit(1)


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), required=True,
              help="Path to the configuration file")
def show_config(config: str):
    """Display a configuration file."""
    try:
        with open(config, "r") as f:
            config_dict = yaml.safe_load(f)
        
        click.echo(yaml.dump(config_dict, default_flow_style=False))
    except Exception as e:
        click.echo(f"Error reading configuration: {e}")
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()