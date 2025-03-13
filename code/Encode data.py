#encoding protein sequence to list of numbers
def encode_seq(sequence: str):
  max_length_sequence = 1000
  sequence_length = len(sequence)
  #check if sequence is correct length
  if sequence_length <= max_length_sequence:
    padding_amount = max_length_sequence - sequence_length
    #add padding when too short
    for i in range(padding_amount):
      sequence = sequence + "Z"
  #take the longest sequence we can
  elif sequence_length > max_length_sequence:
    sequence = sequence[0:max_length_sequence]
  #dictionary to encode sequence NOTE: Z indicates padding
  rules = {"Z":0, "A":1, "R":2, "N":3, "D":4, "C":5, "E":6, "Q":7, "G":8, "H":9, "I":10, "L":11, "K":12, "M":13, "F":14, "P":15, "S":16, "T":17, "W":18, "Y":19, "V":20}
  #encode sequence using our rules dictionary
  encoded_sequence = [rules[amino_acid] for amino_acid in sequence]
  return encoded_sequence
  
    
#testing
test_sequence = "AYVWCE"
print(f"The encoded test sequence is" + str(encode_seq(test_sequence)))
