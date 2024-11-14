def a2m_to_fasta(a2m_file, fasta_file):
    sequences = {}
    with open(a2m_file, 'r') as f:
        current_id = None
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                current_id = line[1:]
                sequences[current_id] = ''
            elif current_id is not None:
                sequences[current_id] += keep_uppercase_letters(line)

    with open(fasta_file, 'w') as f:
        for seq_id, seq in sequences.items():
            f.write(f'>{seq_id}\n{seq}\n')

def keep_uppercase_letters(input_string):
    uppercase_only = ''
    for char in input_string:
        if char.isupper():
            uppercase_only += char
    return uppercase_only

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extract consensus sequence from A2M file and save as FASTA')
    parser.add_argument('--a2m', type=str, help='Input A2M file')
    parser.add_argument('--fasta', type=str, help='Output FASTA file')
    args = parser.parse_args()

    a2m_to_fasta(args.a2m, args.fasta)
