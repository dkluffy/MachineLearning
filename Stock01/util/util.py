import sys
def convert_to_utf8(filename):
    with open(filename) as f:
        l = f.read()
    with open(filename+"_.csv",'wb',) as fw:
        fw.write(l.encode("utf-8"))


if __name__ == "__main__":
    [ convert_to_utf8(f) for f in sys.argv[1:]]