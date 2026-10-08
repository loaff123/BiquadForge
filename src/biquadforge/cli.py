"""Small JSON-reporting command-line interface."""
from __future__ import annotations
import argparse
import json
import sys
from .spec import BiquadForgeError, load_spec
from .qualification import qualify
from .pack import write_pack


def main(argv=None):
    parser=argparse.ArgumentParser(prog="biquadforge",description="Finite-suite scalar Q15 qualification; compiler optional")
    commands=parser.add_subparsers(dest="command",required=True)
    build=commands.add_parser("build");build.add_argument("spec");build.add_argument("--out",required=True)
    verify=commands.add_parser("verify");verify.add_argument("pack");verify.add_argument("--cmsis-root",required=True);verify.add_argument("--cc",default="cc")
    args=parser.parse_args(argv)
    try:
        if args.command=="build":
            result=qualify(load_spec(args.spec));write_pack(result,args.out)
            print(json.dumps({"status":result.status,"out":args.out,"observations":result.observations}))
            return {"rejected":2,"search_budget_exhausted":3}.get(result.status,0)
        from .verify import verify_pack
        result=verify_pack(args.pack,args.cmsis_root,args.cc)
        print(json.dumps(result.as_dict(),sort_keys=True))
        return 0 if result.status=="verified" else 5 if result.status=="compiler_unavailable" else 6
    except BiquadForgeError as e:
        print(json.dumps(e.as_dict()),file=sys.stderr)
        return 4 if e.status=="unsupported" else 1
    except (OSError,ValueError,RuntimeError) as e:
        print(json.dumps({"status":"error","message":str(e)}),file=sys.stderr)
        return 1
