import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)

import click


@click.group(help='MoGe command line interface.')
def cli():
    pass

def main():
    from moge.scripts import app, infer, infer_baseline, infer_panorama, eval_baseline, vis_data
    cli.add_command(app.main, name='app')
    cli.add_command(infer.main, name='infer')
    cli.add_command(infer_baseline.main, name='infer_baseline')
    cli.add_command(infer_panorama.main, name='infer_panorama')
    cli.add_command(eval_baseline.main, name='eval_baseline')
    cli.add_command(vis_data.main, name='vis_data')
    cli()


if __name__ == '__main__':
    main()