
##### Opening and Closing Files ####
f = open('text.txt', 'r')
print(f.name, '\n')
f.close()

##### Reading from files and Using context manager #####
with open('text.txt', 'r') as f:
    pass
# print(f.closed) #returns true

with open('text.txt', 'r') as f:
    #read in entire content of file
    f_contents = f.read()
    print(f_contents)

with open('text.txt', 'r') as f:
    f_contents_list = f.readlines()
    print(f_contents_list)
    print('\n')


with open('text.txt', 'r') as f:
    f_content_one = f.readline()
    print(f_content_one, end='')
    f_content_one = f.readline()
    print(f_content_one, end='')
    print('\n')


#iterate through file
with open('text.txt', 'r') as f:
    for line in f:
        print(line, end="")
    print('\n')

#print out n characters
with open('text.txt', 'r') as f:
    #read in entire content of file
    size_to_read = 6
    f_contents = f.read(size_to_read)

    while len(f_contents) > 0:
        print(f_contents, end="*")
        f_contents = f.read(size_to_read)


#manipulate position of read head
with open('text.txt', 'r') as f:
    #read in entire content of file
    size_to_read = 6
    f_contents = f.read(size_to_read)
    print(f_contents, end="\n")
    print("current position: ", f.tell())

    f.seek(0)
    f_contents = f.read(2*size_to_read)
    print(f_contents, end="\n")
    print("current position: ", f.tell())

    f.seek(15)
    f_contents = f.read(2*size_to_read)
    print(f_contents, end="\n")
    print("current position: ", f.tell())


#### Writing from files #####
with open('test2.txt', 'w') as f:
    f.write('testing') 
    f.write('  testing')
    f.seek(0)
    f.write('I me ')
    

### Read and Write at same time
with open('text.txt', 'r') as rf:
    with open('test2.txt', 'w') as wf:
        for line in rf:
            wf.write(line)


"""with open('pic.jpg', 'rb') as rf:
    with open('pic2.jpg', 'wb') as wf:    
        chunk_size = 4096
        rf_chunk =  rf.read(chunk_size)
        while len(rf_chunk) > 0:
            wf.write(rf_chunk)"""
