from agrifoodpy.pipeline.cli import main
import json
import pytest
import os

def test_cli(tmp_path):

    # Test with no arguments
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 0

    # Argparse help
    with pytest.raises(SystemExit) as e:
        main(['--help'])
    assert e.value.code == 0

    # Process test config file
    script_dir = os.path.dirname(__file__)
    config_filename = os.path.join(script_dir, "data/empty_config.yml")   
    output_filename = str(tmp_path / 'empty.json')
    assert main([config_filename, '-o', output_filename]) == 0

    # Process test config file
    script_dir = os.path.dirname(__file__)
    config_filename = os.path.join(script_dir, "data/test_config.yml")   
    output_filename = str(tmp_path / 'test.json')
    assert main([config_filename, '-o', output_filename]) == 0


def test_cli_load_config(tmp_path, capsys):
    (tmp_path / "sub.yaml").write_text(
        "nodes:\n"
        "  - function: agrifoodpy.utils.nodes.write_to_datablock\n"
        "    name: included\n"
        "    params: {key: result, value: included}\n"
    )
    config = tmp_path / "main.yaml"
    config.write_text(
        "nodes:\n"
        "  - load: sub.yaml\n"
        "  - function: agrifoodpy.utils.nodes.write_to_datablock\n"
        "    name: last\n"
        "    params: {key: result, value: last}\n"
    )
    output = tmp_path / "output.json"

    assert main([str(config), "--nodes", "-o", str(output)]) == 0
    listing = capsys.readouterr().out
    assert listing.index("[0] included") < listing.index("[1] last")
    assert json.loads(output.read_text()) == {"result": "last"}
