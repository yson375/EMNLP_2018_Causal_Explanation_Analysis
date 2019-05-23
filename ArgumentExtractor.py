import csv
import sys

# Find the disocurse connectives with the given length, extract discoruse arguments using them
def disconMarking(mwc_conns, arg, forms, u_poss, head_idxs, connective_list, temp_args, disConLength):
    prev_p_marker=arg[0]
    mwc_found=False
    i=arg[0]



    while i < arg[1]-disConLength: # the range should be (start, end-3) because 'on the one hand' + Parg2

        if i in mwc_conns: # skip when the index falls in to the mwc already found
            i += 1
            continue
        words=' '.join([str(forms[j]) for j in range(i,i+disConLength)])
        

        if words in connective_list:
            if ('L' in u_poss[prev_p_marker:i] or 'M' in u_poss[prev_p_marker:i] or 'V' in u_poss[prev_p_marker:i] \
                or 'Y' in u_poss[prev_p_marker:i] or 'G' in u_poss[prev_p_marker:i]) \
                    and ('L' in u_poss[i+disConLength:arg[1]] or 'M' in u_poss[i+disConLength:arg[1]] or 'V' in u_poss[i+disConLength:arg[1]] \
                         or 'Y' in u_poss[i+disConLength:arg[1]] or 'G' in u_poss[i+disConLength:arg[1]]):
                temp_args.append((prev_p_marker,i+disConLength))
                for j in range(disConLength):
                    mwc_conns.add(i+j)
                mwc_found=True
                prev_p_marker=i+disConLength
                i=i+disConLength
                continue
        i+=1

    if mwc_found:
        if prev_p_marker!=arg[1]:
            temp_args.append((prev_p_marker, arg[1]))
    else:
        temp_args.append((arg[0],arg[1]))


def parg_extraction(arg, forms, u_poss, head_idxs, connective_list, final_arg_list):

    # multi-word connectives filtering
    # finding 4-word length connectives (the longest PDTB connective)
    prev_s_marker = arg[0]
    mwc_conns=set()
    mwc_found=False
    i=arg[0]

    mwc_pargs=[]
    disconMarking(mwc_conns, arg, forms, u_poss, head_idxs, connective_list, mwc_pargs, 4)


    # finding 3-word length connectives
    for disConLength in range(3,1,-1):
        temp_pargs = []
        for mwc_parg in mwc_pargs:
            disconMarking(mwc_conns, mwc_parg, forms, u_poss, head_idxs, connective_list, temp_pargs, disConLength)
        mwc_pargs=temp_pargs


    # tweet-specific '&' tags

    i = arg[0]

    for mwc_parg in mwc_pargs:
        prev_s_marker = mwc_parg[0]
        found_connective = False
        while i <mwc_parg[1]:
            if i in mwc_conns: # skip when the index falls in to the mwc already found
                i += 1
                continue
            if forms[i].lower() in connective_list or u_poss[i] =='&' or u_poss[i] =='P' or (forms[i]==',' and u_poss[i]==','):
                if ('L' in u_poss[prev_s_marker:i] or 'M' in u_poss[prev_s_marker:i] or 'V' in u_poss[prev_s_marker:i] \
                    or 'Y' in u_poss[prev_s_marker:i] or 'G' in u_poss[prev_s_marker:i]) \
                    and ('L' in u_poss[i:mwc_parg[1]] or 'M' in u_poss[i:mwc_parg[1]] or 'V' in u_poss[i:mwc_parg[1]] \
                    or 'Y' in u_poss[i:mwc_parg[1]] or 'G' in u_poss[i:mwc_parg[1]]):
                    final_arg_list.append((prev_s_marker,i+1))
                    prev_s_marker=i+1
                    found_connective=True
            i += 1

        if found_connective:
            if prev_s_marker!=mwc_parg[1]:
                final_arg_list.append((prev_s_marker, mwc_parg[1]))
        else:
            final_arg_list.append((mwc_parg[0],mwc_parg[1]))



def arg_extraction(idxs, forms, u_poss, head_idxs, connective_list):
    # Sentence extraction
    s_markers = []  # indexes for boudnaries sentences
    twt_len=len(idxs)
    prev_s_marker=0
    for i in range(len(idxs)):
        if u_poss[i] == 'E':
            if i!=0:
                s_markers.append(i)
            if i!=twt_len-1:
                s_markers.append(i + 1)
            # base case E on the first idx or the last idx
            prev_s_marker=i+1
        elif u_poss[i] == ',' and (len(forms[i])>1 or forms[i]=='.' or forms[i]=='!'): # take '.' or '!!!' as a sentence demarcator
            if prev_s_marker==i: # append this this token to the previous sentence if this is a consecutive marker
                if len(s_markers)==0:
                    s_markers.append(i+1)
                else:
                    s_markers[len(s_markers)-1]=i+1
            else:
                s_markers.append(i + 1)
            prev_s_marker = i + 1
    sarg_idxs=[]
    prev_s_marker=0
    for i in s_markers:
        sarg_idxs.append((prev_s_marker,i)) #(start index, end index+1)
        prev_s_marker=i
    if len(s_markers)==0: # when there are no sentences in a tweet (one-sentence tweet)
        sarg_idxs.append((0, twt_len)) # make it as one sentence arg
    else:
        if sarg_idxs[len(sarg_idxs)-1][1] != twt_len: # if the parser didn't capture the last senten #sarg_idxs=[sarg for sarg in sarg_idxs if sarg[0] == sarg[1]] # remove empty dus which fall into the base cases
            sarg_idxs.append((sarg_idxs[len(sarg_idxs) - 1][1], twt_len))
    final_arg_list=[]

    for sarg in sarg_idxs:
        if sarg[0] == sarg[1]: #edge cases where sentence marker 'E' and '.' was overlapped
            continue
        parg_extraction(sarg, forms, u_poss, head_idxs, connective_list, final_arg_list)

    return final_arg_list

if len(sys.argv) != 2:
    print("USAGE> ArgumentExtractor.py [tweebo_output.predict]")
    sys.exit()
tweebo_file=open(sys.argv[1],'r')
connective_list = ['accordingly','additionally','after','afterward','afterwards','also','alternatively','although','and','as','as a result','as an alternative','as if','as long as','as soon as','as though','as well','because','before','before and after','besides','but','by comparison','by contrast','by then','consequently','conversely','earlier','either','else','except','finally','for','for example','for instance','further','furthermore','hence','however','if','if and when','then','in addition','in contrast','in fact','in other words','in particular','in short','in sum','in the end','in turn','indeed','insofar as','instead','later','lest','likewise','meantime','meanwhile','moreover','much as','neither','nevertheless','next','nonetheless','nor','now that','on the contrary','on the one hand','on the other hand','once','or','otherwise','overall','plus','previously','rather','regardless','separately','similarly','simultaneously','since','so','so that','specifically','still','then','thereafter','thereby','therefore','though','thus','till','ultimately','unless','until','when','when and if','whereas','while','yet']
connective_list = set([u''+connective for connective in connective_list])

output_file=open(sys.argv[1]+'.args.csv','w')
output_csv=csv.writer(output_file)
twt_id=1

output_csv.writerow(['tweet_id','arg_id','message'])
sentences=[]
idxs=[]
forms=[]
u_poss=[]
head_idxs=[]
t_pos=[]
a_pos=[]
for line in tweebo_file:
    if line=='\n': # new line detected: new tweet parsing result
        # (1) sentence extraction: segmenting with ',', 'E'
        twt_arg_idxs=arg_extraction(idxs, forms, u_poss, head_idxs, connective_list)

        u_pos_str=''
        for u_pos in u_poss:
            u_pos_str+=' '+u_pos
        #tester_code (1) if arg idxs are always connected, (2) if there are duplicates
        arg_start_idxs=[]
        arg_end_idxs=[]
        dup_found=False
        discon_found=False
        prev_end_idx=0
        for i in range(len(twt_arg_idxs)):
            arg_idx=twt_arg_idxs[i]

            if prev_end_idx!=arg_idx[0]:
                discon_found=True
            prev_end_idx=arg_idx[1]

            if arg_idx[0] in arg_start_idxs:
                dup_found=True
            arg_start_idxs.append(arg_idx[0])

            if arg_idx[1] in arg_end_idxs:
                dup_found=True
            arg_end_idxs.append(arg_idx[1])
            arg_form_str=''
            pos_str=''
            arg_id=str(twt_id) + '-' + str(i)

            for i in range(arg_idx[0],arg_idx[1]):
                arg_form_str+=' '+str(forms[i])
                pos_str+=' '+(u_poss[i])
            output_csv.writerow([twt_id,arg_id,arg_form_str[1:]])



        #after extraction,  increment twt_id
        twt_id+=1
        idxs = []
        forms = []
        u_poss = []
        head_idxs = []
        sentences=[]
        s_ptr=0 # index after period

    else:
        tokens=line.split('\t')
        idxs.append(int(tokens[0]))
        forms.append(tokens[1])
        lemma=tokens[2]
        u_poss.append(tokens[3])
        s_pos=tokens[4]
        feat=tokens[5]
        head_idxs.append(int(tokens[6]))
        dep_rel=tokens[7]


tweebo_file.close()
output_file.close()





