def neuron(x1, x2, w1, w2, w0):
    s = w1 * x1 + w2 * x2 + w0
    return 1 if s > 0 else 0

def OR(x1, x2):
    # розділяюча лінія x1 + x2 = 1/2  ->  w1=1, w2=1, w0=-1/2
    return neuron(x1, x2, 1, 1, -0.5)

def AND(x1, x2):
    # розділяюча лінія x1 + x2 = 3/2  ->  w1=1, w2=1, w0=-3/2
    return neuron(x1, x2, 1, 1, -1.5)

def XOR(x1, x2):
    y1 = OR(x1, x2)
    y2 = AND(x1, x2)
    # у просторі (y1, y2) класи (0,0)->1 і (1,1)->1 розділяє пряма y1 + y2 = 1/2 (NOT AND(y1,y2)? )
    # фактично XOR(x1,x2) = AND(y1, NOT y2) = AND(OR(x1,x2), NOT AND(x1,x2))
    return neuron(y1, y2, 1, -1, -0.5)

if __name__ == "__main__":
    print(f"{'x1':^4}{'x2':^4}{'OR':^4}{'AND':^5}{'XOR(my)':^9}{'XOR(true)':^10}{'match':^7}")
    all_ok = True
    for x1 in (0, 1):
        for x2 in (0, 1):
            true_xor = x1 ^ x2
            my_xor = XOR(x1, x2)
            ok = (my_xor == true_xor)
            all_ok &= ok
            print(f"{x1:^4}{x2:^4}{OR(x1,x2):^4}{AND(x1,x2):^5}{my_xor:^9}{true_xor:^10}{str(ok):^7}")
    print("\nУсі 4 комбінації співпадають з істинною функцією XOR:", bool(all_ok))
