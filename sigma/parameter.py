Atom_radius = {'N' :71, 'Se':116, 'F':64, 'Co':111, 'O':63,'As':121,'Br':114,'Cl':99,  'S':103,'C' :75, 'P':111, 'I':133,'H':32}
Atom_mass = {'N':14.00674,'Se':78.96,'F':18.9984032,'Co':58.933195,'As':74.92160,'O':15.9994,'Br':79.904,'Cl':35.453,'S':32.065,'C':12.0107,'P':30.973762,'I':126.90447,'H':1.00794}

class Parameter:
    '''
    * Constants used within the code
    *
    * Attributes
    * ----------
    * adduct_SET : Adduct Set
    * All_Atoms  : A Set of all elements in a SMILES dataset
    * Max_Coor   : The maximum value in all coordinate data
    * Min_Coor   : The minimum value in all coordinate data
    '''
    def __init__(self, adduct_SET=[], All_Atoms=[], Max_Coor=0, Min_Coor=0, **kwargs):
        self.adduct_SET = adduct_SET
        self.All_Atoms  = All_Atoms
        self.Max_Coor = Max_Coor
        self.Min_Coor = Min_Coor
        super().__init__(**kwargs)