"""
This is version 1.0
"""

basic = (
    "\u0627\u0628\u062A\u062B\u062C\u062D\u062E\u062F\u0630\u0631\u0632\u0633\u0634"
    "\u0635\u0636\u0637\u0638\u0639\u063A\u0641\u0642\u0643\u0644\u0645\u0646\u0647\u0648\u064A"
)
extension = "\u0622\u0629\u0640\u0649"
hamza = "\u0621\u0623\u0624\u0625\u0626"
harakat = "\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652"

basic_buckwalter = "AbtvjHxd*rzs$SDTZEgfqklmnhwy"
extension_buckwalter = "|p%Y"
hamza_buckwalter = "'>&<}"
harakat_buckwalter = "FNKaui+o"

special = "_^~ "
punctuation = '.,!?:;-"()'
numbers = "0123456789"

arabic_dictionary = special + basic + extension + hamza + harakat + numbers + punctuation

buckwalter_dictionary = (
    special + basic_buckwalter + extension_buckwalter + hamza_buckwalter + harakat_buckwalter + numbers + punctuation
)

arabic_index_dict = {}
for i, char in enumerate(arabic_dictionary):
    arabic_index_dict[char] = i + 1

arabic_inverse_index_dict = {v: k for k, v in arabic_index_dict.items()}

harakat_index_dict = {"_": 0}
for i, char in enumerate(harakat):
    harakat_index_dict[char] = i + 1

if __name__ == "__main__":
    print(len(arabic_dictionary))
