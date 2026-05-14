from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


def test_init_scripts_start_rgb_only_camera_launch():
    init_sh = (ROOT / "scripts" / "init.sh").read_text()
    init_headless_sh = (ROOT / "scripts" / "init_headless.sh").read_text()

    assert "scripts/launch/multi_camera_rgb.launch" in init_sh
    assert "scripts/launch/multi_camera_rgb.launch" in init_headless_sh
    assert "roslaunch astra_camera multi_camera.launch" not in init_sh
    assert "roslaunch astra_camera multi_camera.launch" not in init_headless_sh


def test_init_headless_sh_is_headless_and_does_not_open_viewers():
    init_headless_sh = (ROOT / "scripts" / "init_headless.sh").read_text()

    assert "gnome-terminal" not in init_headless_sh
    assert "rqt_image_view" not in init_headless_sh


def test_init_headless_sh_dry_run_is_executable_without_starting_hardware():
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "init_headless.sh"), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "DRY-RUN" in result.stdout
    assert "setsid nohup roslaunch" in result.stdout
    assert "multi_camera_rgb.launch" in result.stdout
    assert "can_config.sh" in result.stdout


def test_init_headless_sh_can_start_depth_without_point_cloud():
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "init_headless.sh"),
            "--dry-run",
            "--with-depth",
            "--no-can",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "enable_depth:=true" in result.stdout
    assert "enable_point_cloud:=false" in result.stdout


def test_rgb_camera_launch_disables_depth_and_point_cloud():
    launch_path = ROOT / "scripts" / "launch" / "multi_camera_rgb.launch"
    root = ET.parse(launch_path).getroot()

    top_level_args = {
        arg.attrib["name"]: arg.attrib["default"]
        for arg in root.findall("arg")
        if "name" in arg.attrib and "default" in arg.attrib
    }
    assert top_level_args["enable_depth"] == "false"
    assert top_level_args["enable_point_cloud"] == "false"
    assert top_level_args["depth_align"] == "false"

    includes = root.findall("include")
    assert len(includes) == 3
    for include in includes:
        include_args = {
            arg.attrib["name"]: arg.attrib["value"]
            for arg in include.findall("arg")
            if "name" in arg.attrib and "value" in arg.attrib
        }
        assert include_args["enable_depth"] == "$(arg enable_depth)"
        assert include_args["enable_point_cloud"] == "$(arg enable_point_cloud)"
        assert include_args["depth_align"] == "$(arg depth_align)"
