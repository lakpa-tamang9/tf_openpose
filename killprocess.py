import psutil

# For multiple process
# myProcess = psutil.pids()[-10:]
# procName = []
# for pro in myProcess:  
#     p = psutil.Process(pro)
#     procName.append(p.name())
# print(procName)

# For single process
highProcess = psutil.pids()[-2:]
p_0 = psutil.Process(highProcess[0])
p_1 = psutil.Process(highProcess[1])
# print(p_0.name)
p_0.terminate()
p_1.terminate()
