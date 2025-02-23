FILENAME = "popularity.csv"

def get_dict( ):
    """
    : return: all_d (dict{str: dict{str: int}}): Dictionary holding sex as key,
    name_score_d as value.
    (name_score_d holds name as key, rank as value)

    """
    all_d = {}
    female_score_d = {}
    male_score_d = {}

    with open(FILENAME, "r", encoding="utf-8") as file:
        first_line = file.readline().strip()  # 讀取第一行並去除換行字元
        #print(first_line)
        first_line_list = first_line.split(',')
        #print(first_line_list)
        total_item_in_a_row = len(first_line_list)
        max_score = (total_item_in_a_row - 1)//2
        #print("total_item_in_a_row : " + str(total_item_in_a_row))
        #print("max_score : " + str(max_score))

        file.seek(0)

        for line in file:  
            female_score_counter = max_score
            male_score_counter = max_score

            #print(line.strip())  
            line_list = [itme.strip() for itme in line.split(',')]

            for i in range(len(line_list)):
                if i == 0:
                    continue
                # female case
                if i <= max_score:
                    score = female_score_d.setdefault(line_list[i],0) 
                    #print("score : {0}".format(score) )
                    #print("line_list[i] : {0}".format(line_list[i]) )
                    #print(" female_score_d[ {0} ] += {1}".format(line_list[i], female_score_counter) )
                    female_score_d[ line_list[i] ] += female_score_counter
                    female_score_counter -= 1
                # male case
                elif i <= (2*max_score):
                    score = male_score_d.setdefault(line_list[i],0) 
                    male_score_d[ line_list[i] ] += male_score_counter
                    male_score_counter -= 1
       

    all_d["female"] = female_score_d
    all_d["male"] = male_score_d


    print("all_d : {0}".format(all_d) )

get_dict( )
