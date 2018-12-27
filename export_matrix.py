import csv
import dataset_info as di

def print_matrix(matrix):
    print('--------------------------------------------------------')
    for i,row in enumerate(matrix):
        print('{:>6}|'.format(di.label2name(i)), end='')
        if int(sum(row)) is not 0:
            for item in row:
                print('{:>7.2%}'.format(item/sum(row)), end='')
            print('|{:>6}|{:.2%}'.format(sum(row),matrix[i][i]/sum(row)))
        else:
            for item in row:
                print('{:>7.2%}'.format(item), end='')
            print('|{:>6}|{:.2%}'.format(sum(row),matrix[i][i]))

def save_matrix(matrix,path,filename,epoch,acc,raw=None):
    with open(path+filename, 'w') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        writer.writerow(['epoch',epoch,'total accuracy',acc])
        writer.writerow(['#']+di.names()+['total','accuracy'])
        for i,row in enumerate(matrix):
            if int(sum(row)) is not 0:
                writer.writerow([di.label2name(i)]+(row/sum(row)).tolist()+[sum(row),row[i]/sum(row)])
            else:
                writer.writerow([di.label2name(i)]+row.tolist()+[sum(row),0])
    if raw is not None:
        with open(path+'raw/'+filename, 'w') as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n')
            writer.writerow(['epoch',epoch,'total accuracy',acc])
            writer.writerow(['#']+di.names()+['total','accuracy'])
            
            for i,row in enumerate(matrix):
                if int(sum(row)) is not 0:
                    writer.writerow([di.label2name(i)]+row.tolist()+[sum(row),row[i]/sum(row)])
                else:
                    writer.writerow([di.label2name(i)]+row.tolist()+[sum(row),0])
