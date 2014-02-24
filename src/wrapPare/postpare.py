def saveHistory(filename, history):
    content = history.dump()
    with open(filename, 'w') as ifile:
        ifile.write(" ".join(history.states.keys()) + "\n")
        ifile.write("pt, angle, v\n")
        ifile.write(content)