"""
Review python os module
"""
import os

#see all attributes & methods of this module
#print(dir(os))

# get current working directory
print(os.getcwd())
# navigate to new location
os.chdir('/Users/danielobembe/Desktop')
print(os.getcwd())
# os.chdir('/Users/danielobembe/Projects/research_notes/misc')

#list files in current direcoty
print(os.listdir())

# makde directorys
os.mkdir('os_temp_demo_2')
os.makedirs('os_temp_demo_1/example_dir')
print(os.listdir())

# remove directorys
os.rmdir('os_temp_demo_2')
os.removedirs('os_temp_demo_1/example_dir')

