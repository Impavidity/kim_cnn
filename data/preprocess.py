from utils import clean_str_sst

def process(file_in, file_out):
    print("Currently processing file {}".format(file_in))
    fin = open(file_in, "r")
    fout = open(file_out, "w")
    for line in fin.readlines():
        label = line[0]
        sent = clean_str_sst(line[2:])
        fout.write("{}\t{}\n".format(label, sent))
    fin.close()
    fout.close()
    print("Save file to {}".format(file_out))

        


if __name__=="__main__":
    process("stsa.fine.phrases.train.tsv", "stsa.fine.phrases.train")
    process("stsa.fine.dev.tsv", "stsa.fine.dev")
    process("stsa.fine.test.tsv", "stsa.fine.test")
