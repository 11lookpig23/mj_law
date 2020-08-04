# Scheme heavily adapted from https://github.com/deepmind/pycolab/
# '@' means "wall"
# 'P' means "player" spawn point
# 'A' means appl2 spawn point
# '' is empty space
##  38x16
##  28x12
HARVEST_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@ P   P      A    P AAAAA    P  A P  @',
    '@  P     A P AA    P    AAA    A  A  @',
    '@     A AAA  AAA    A    A AA AAAA   @',
    '@ A  AAA A    A  A AAA  A  A   A A   @',
    '@AAA  A A    A  AAA A  AAA        A P@',
    '@ A A  AAA  AAA  A A    A AA   AA AA @',
    '@  A A  AAA    A A  AAA    AAA  A    @',
    '@   AAA  A      AAA  A    AAAA       @',
    '@ P  A       A  A AAA    A  A      P @',
    '@A  AAA  A  A  AAA A    AAAA     P   @',
    '@    A A   AAA  A A      A AA   A  P @',
    '@     AAA   A A  AAA      AA   AAA P @',
    '@ A    A     AAA  A  P          A    @',
    '@       P     A         P  P P     P @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']

HARVEST_MAP2 = [
                         
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@ P   P    A    P   P  A P  @',
    '@  P   A P AA    P    A  A  @',
    '@     AAA  AAA    AA AAAA   @',
    '@ A  AAA    A  A AA   A A   @',
    '@AAA  A    A  AAA        A P@',
    '@ A A  A  AAA  A AA   AA AA @',
    '@  A A AA    A A  AAA  A    @',
    '@     A   A A  AAA A  AAA P @', 
    '@ A   A A  AAA  A      A    @',
    '@     A     A       P  P  P @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']

HARVEST_MAP3 = [                        
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@ A   A    A    A   A  A A  @',
    '@  A   A P AA    P    A  A  @',
    '@    PAAA  AAA P  AA AAAA   @',
    '@ A  AAA  P A  A AA P A A   @',
    '@AAA  A  P A  AAA  A  P  A P@',
    '@ A A  A  AAA  A AA P AA AA @',
    '@  A A AA  P A A  AAA  A    @',
    '@     A   A A  AAA A  AAA A @', 
    '@ A   A A  AAA  AP  P A  A  @',
    '@     A     A       A  A  A @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']
HARVEST_MAP4 = [                        
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@ A   A    A    A   A  A A  @',
    '@  A   A A AA    A    A  A  @',
    '@    PAAA  AAA P  AA AAAA   @',
    '@ A  AAA  P A  A AA P A A   @',
    '@AAA  A  P A  AAA  A  P  A A@',
    '@ A A  A  AAA  A AA P AA AA @',
    '@  A A AA  P A A  AAA  A    @',
    '@     A   A A  AAA A  AAA A @', 
    '@ A   A A  AAA  AA    A  A  @',
    '@     A     A       A  A  A @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']

HARVEST_MAP5 = [                        
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@ A   A    A    A   A  A A  @',
    '@  A   A   AA    A  AAA  A  @',
    '@     AAA   AA P     A AA   @',
    '@ A  AAA  P A  A  A P A A   @',
    '@ AA  A  P A  A P  A  P  A A@',
    '@ A A  A  AP   P A  P AA AA @',
    '@ AA A AA  P A A   AA   A   @',
    '@   A A   A A   AA A   AA A @', 
    '@ A   A A  AAA  AA    A  A  @',
    '@     A     A    A  A     A @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']

HARVEST_MAP6 = [                        
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@ A   A    A    A   A  A A  @',
    '@  AA AA   AA P  A   AAA A  @',
    '@  A  AAA   AA     A   AA   @',
    '@ A   A   P A AA A  P A A   @',
    '@     A  P    A P  AA P  A A@',
    '@ A    A   P  AA P  P A  A  @',
    '@  AA  AA  P A A   A  AA    @',
    '@ A A A   A        A   AA A @', 
    '@ A  AA A    A  AA    AA  A @',
    '@     A     AA    A    A  A @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']


HARVEST_MAP7 = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@ P   P      A    P AAAAA    P  A P  @',
    '@  P     A P AA    P    AAA    A  A  @',
    '@     A AAA  AAA    A    A AA AAAA   @',
    '@ A  AAA A    A  A AAA  A  A   A A   @',
    '@AAA  A A    A  AAA A  AAA        A P@',
    '@ A A  AAA  AAA  A A    A AA   AA AA @',
    '@  A A  AAA    A A  AAA    AAA  A    @',
    '@   AAA  A      AAA  A    AAAA       @',
    '@ P  A       A  A AAA    A  A      P @',
    '@A  AAA  A  A  AAA A    AAAA     P   @',
    '@    A A   AAA  A A      A AA   A  P @',
    '@     AAA   A A  AAA      AA   AAA P @',
    '@ A    A     AAA  A  P          A    @',
    '@       P     A         P  P P     P @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']
CLEANUP_MAP = [
    '@@@@@@@@@@@@@@@@@@',
    '@RRRRRR     BBBBB@',
    '@HHHHHH      BBBB@',
    '@RRRRRR     BBBBB@',
    '@RRRRR  P    BBBB@',
    '@RRRRR    P BBBBB@',
    '@HHHHH       BBBB@',
    '@RRRRR      BBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@RRRRR   P P BBBB@',
    '@HHHHH   P  BBBBB@',
    '@RRRRRR    P BBBB@',
    '@HHHHHH P   BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH    P  BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHHH  P P BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHHH      BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@@@@@@@@@@@@@@@@@@']
