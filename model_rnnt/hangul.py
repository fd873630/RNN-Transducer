

chosung = ("ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ")

jungsung = ("ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ")

jongsung = ("", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ")

def isHangeul(one_character):
    return 0xAC00 <= ord(one_character[:1]) <= 0xD7A3

def hangeulExplode(one_hangeul):
    a = one_hangeul[:1]
    if isHangeul(a) != True:
        return False
    b = ord(a) - 0xAC00
    cho = b // (21*28)
    jung = b % (21*28) // 28
    jong = b % 28
    if jong == 0:
        return (chosung[cho], jungsung[jung])
    else:
        return (chosung[cho], jungsung[jung], jongsung[jong])

def hangeulJoin(inputlist):
    result = ""
    cho, jung, jong = 0, 0, 0
    inputlist.insert(0, "")
    while len(inputlist) > 1:
        if inputlist[-1] in jongsung:
            if inputlist[-2] in jungsung:
                jong = jongsung.index(inputlist.pop())
            
            else:
                result += inputlist.pop()
        elif inputlist[-1] in jungsung:
            if inputlist[-2] in chosung:
                jung = jungsung.index(inputlist.pop())
                cho = chosung.index(inputlist.pop())
                result += chr(0xAC00 + ((cho*21)+jung)*28+jong)
                cho, jung, jong = 0, 0, 0
            else:
                result += inputlist.pop()

        else:
            result += inputlist.pop()
    else:
        return result[::-1]

def pureosseugi(inputtext):
    result = ""
    for i in inputtext:
        if isHangeul(i) == True:
            for j in hangeulExplode(i):
                result += j
        else:
            result += i
    
    return result

def moasseugi(inputtext):
    t1 = []
    for i in inputtext:
        t1.append(i)

    return hangeulJoin(t1)
