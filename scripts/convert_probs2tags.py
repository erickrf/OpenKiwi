import argparse

"""
Script to convert tag probabilities to discrete labels.

Values lesser or equal to 0.5 are converted to OK,
values greater than 0.5 are converted to BAD.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('-o', dest='output',
                        help='If not given, will be the same as input')
    args = parser.parse_args()

    output_path = args.output if args.output is not None else args.input

    with open(args.input, 'r') as f:
        lines = f.read().splitlines()

    with open(output_path, 'w') as f:
        for line in lines:
            probs = [float(value) for value in line.split()]
            tags = ['BAD' if prob > 0.5 else 'OK'
                    for prob in probs]
            output_line = ' '.join(tags) + '\n'
            f.write(output_line)
