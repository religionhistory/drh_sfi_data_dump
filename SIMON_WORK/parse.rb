require 'csv'

# 21 questions
# 390 groups
file=File.new("../supreme_high_gods.csv", 'r'); ans=CSV.parse(file.read); file.close

qhash=Hash.new

religion=Hash.new
entry_hash=Hash.new
ans[1..-1].each { |i|
  entry=i[13]
  entry_hash[i[13]]=i[12].gsub(/[^A-Za-z\ 0-9\-]/, "")
  date_range=i[15].gsub(" ", "")
  region=i[22]
  sector=i[18]
  name="#{entry}ENTRY_#{date_range}_#{region}REGION"
  if i[3].include?("Group") and i[9] == "Yes" and i[0] != "4960" and i[0] != "4835" and i[0] != "4857" and sector.include?("Non-elite") then
    if religion[name] == nil then
      religion[name]=Hash.new([])
    end
    religion[name][i[0]] += [i[6]]
  end
  qhash[i[0]]=i[1]
};1
qlist=religion.keys.collect { |i| religion[i].keys }.flatten; qlist=(qlist & qlist).sort
religion.keys.each { |i|
  (qlist-religion[i].keys).each { |j|
    religion[i][j]=["No answer at all"]
  }
}

### JUST TAKING ONE ROW FOR EACH...
religion.keys.each { |i|
  religion[i].keys.each { |j|
    religion[i][j]=religion[i][j][0]
  }
}

file=File.new("../SIMON_PROCESSED/processed_supreme_high_gods.csv", 'w')
str="Entry ID,Religion Name,Region ID,Date Range,"+qlist.collect { |i| i }.join(",")+"\n"
religion.keys.each { |i|
  entry=i.split("_")[0].gsub(/[^0-9]/,"")
  region=i.split("_")[2].gsub(/[^0-9]/,"")
  date=i.split("_")[1]
  str += "#{entry},#{entry_hash[entry]},#{region},#{date},"
  str += religion[i].keys.collect { |j|
    religion[i][j] == "Yes" ? "1" : (religion[i][j] == "No" ? "0" : "X")
  }.join(",")+"\n"
}
file.write(str); file.close

religion.keys.select { |i|
  qlist.collect { |j| religion[i][j] == "Yes" ? "1" : (religion[i][j] == "No" ? "0" : "X") }.join("").scan(/X/).length > 5
}.each { |i|
  religion.delete(i)
}

file=File.new("supreme_high_gods.dat", 'w')
file.write("#{religion.keys.length}\n#{qlist.length}\n")
religion.keys.each { |i|
  file.write(qlist.collect { |j| religion[i][j] == "Yes" ? "1" : (religion[i][j] == "No" ? "0" : "X") }.join("")+" 1.0\n")
}
file.close

# `/Users/simon/Desktop/humanities-glass/MPF_CMU/mpf -c supreme_high_gods.dat 1.0`

params_neghalf=[7.2120216019e-09, -3.3539216379e-01, 1.6657075990e-01, -2.6304824788e-01, 6.1906601400e-02, -2.0516200355e-01, 9.5849292039e-08, 1.1604318115e-01, -4.9566083740e-07, 2.9759242896e-01, 3.0710855142e-01, 2.9086155817e-01, 1.8865009399e-01, 3.5802823105e-02, 5.0983603113e-01, -3.3444980372e-02, 2.8537259590e-01, -1.7738617409e-08, 3.5637865218e-02, 2.4930207609e-02, 4.4275155178e-01, 5.8481058354e-02, 1.2430287043e-01, 2.5269116370e-01, 1.4180809554e-01, 2.3257945057e-02, -2.1477853827e-01, 4.2277148908e-01, 1.2903140069e-01, 3.6803217357e-01, -2.2780576310e-01, -3.1870406315e-01, -2.6966296451e-01, 2.5889567804e-02, 1.1445833173e-01, 4.8516064486e-02, 3.1247203635e-01, -2.9773505051e-01, -1.1700063137e-01, 8.6002551742e-01, 3.7032225397e-08, 3.6667635907e-08, -4.4138606122e-01, 6.4048812545e-10, 2.6058676745e-02, 4.5625802557e-09, -1.4591290145e-01, -9.3461106516e-02, 3.0768112539e-03, 3.2867603374e-08, 9.8972515664e-02, -7.6275995367e-02, -1.6327596351e-01, -2.2397805687e-01, 6.7906456880e-02, 1.6293316124e-08, 1.7029243412e-01, -5.4979410757e-02, 1.9664247330e-01, 5.8491030074e-02, -2.5493195673e-03, -4.6827514837e-01, 1.6588931721e-08, 8.6649702421e-04, -5.0273042923e-01, 1.1639196978e-01, 2.8993705678e-01, 3.4081784767e-01, -3.9193378835e-03, 7.3260154821e-01, -1.2986208371e-01, 1.5432357620e-02, 7.2686454356e-01, -2.7656765679e-01, -1.5443670908e-01, -8.3324851502e-01, -4.1923812787e-08, 9.4255149809e-01, 6.0064407694e-01, 1.7324211548e-02, -7.9821573215e-08, -4.3033208101e-01, 2.5738801520e-01, 9.5241231729e-01, 2.2576083551e-07, 5.0568814035e-01, -2.4320575732e-02, 7.1850328899e-09, -3.7089096926e-01, 3.4793817627e-01, -1.9777254442e-09, 2.4542244965e-01, 4.7796624590e-08, 4.3405001746e-01, -5.2594720400e-08, 1.1216326894e+00, 2.6361197477e-08, 3.3979774116e-01, 5.6909200235e-07, 6.0890832715e-01, -3.4417722225e-01, 7.2431626960e-01, 5.2335439867e-01, 7.1199378618e-02, 6.2265716954e-02, -7.8301775200e-08, 1.2719557758e-02, -1.4791614574e-07, -1.3502992653e-07, -6.7180018995e-08, -3.2014651798e-02, 8.6220900670e-08, 1.1578878971e-01, 1.0246255917e-07, 3.8953115888e-01, 7.3174915513e-08, -2.2406053655e-01, -3.3710561707e-01, 4.6460900785e-01, 5.5476503557e-08]

n=15
count=0
lookup=Hash.new
0.upto(n-2) { |i|
  (i+1).upto(n-1) { |j|
    lookup[[i,j]]=params_neghalf[count]
    count += 1
  }
}
n.times { |i|
  lookup[i]=params_neghalf[count+i]
}

lookup.keys.select { |i|
  i.class != Array
}.sort { |i,j| lookup[j].abs <=> lookup[i].abs }[0..19].each { |i|
  print "#{i} (#{qhash[qlist[i]]}: #{lookup[i]}\n"
}


lookup.keys.select { |i|
  i.class == Array
}.sort { |i,j| lookup[j].abs <=> lookup[i].abs }[0..19].each { |i|
  print "#{i} (#{qhash[qlist[i[0]]]} and #{qhash[qlist[i[1]]]}): #{lookup[i]}\n"
}

lookup.keys.select { |i|
  i.class == Array
}.sort { |i,j| lookup[j].abs <=> lookup[i].abs }.reverse[0..19].each { |i|
  print "#{i} (#{qhash[qlist[i[0]]]} and #{qhash[qlist[i[1]]]}): #{lookup[i]}\n"
}




## vv OLDER

# sbatch -N 1 -o SFI_TESTS -t 12:00:00 -p RM mon.sh
params_neghalf=[-2.3092266858e-06, -5.8913274785e-05, 1.4211675564e-01, -5.5595353624e-06, 1.6333902281e-01, 6.0566843282e-01, 9.9593009999e-06, -1.8645706118e-06, 8.2410720701e-02, 2.8738479868e-01, -1.0221541012e-02, 3.1227557485e-01, -1.8779092525e-07, 1.9817110878e-01, 2.5834003698e-06, 1.1449580714e-01, -5.5498327120e-02, -5.2710579603e-01, -2.7954301859e-01, 1.4161296913e-02, 2.7167942195e-06, -1.3612360798e-01, 2.7578286797e-02, 6.0041224510e-07, 1.0885941827e-01, -4.6477214475e-01, 1.7407786003e-01, -5.4630834000e-02, -5.4884998786e-02, 3.2452080816e-01, -1.4276137363e-02, 4.6959412651e-01, 3.3531866206e-01, 2.8198932075e-07, 1.7649579915e-01, -6.3152253847e-02, 8.4416376499e-02, 9.7350560332e-01, 7.0568315332e-01, 1.9797344572e-01, 2.4423569331e-01, 5.6872075699e-01, 5.9499313640e-06, -3.7623690091e-06, 3.1129829785e-03, -1.5955916917e-01, 2.5589016461e-01, 1.3481770570e-01, 8.5914573251e-02, 2.7120490693e-01, 3.8459236549e-02, 6.9358453659e-02, 4.1703268789e-06, 1.5953462208e-01, 9.4160825749e-01, 1.1026998070e-01, 1.8198264245e-02, 1.9079437114e-01, -2.3161597793e-01, 4.3031847385e-02, 1.4069717443e-01, -2.0109705630e-01, 6.5732227472e-06, 1.7385031890e-01, 2.7590094455e-06, -1.0671161460e-01, -3.4006244069e-01, 1.8026477665e-01, 3.1698037061e-01, -6.2432243599e-02, 1.3010075298e-01, -6.5944279601e-02, 1.3261267321e-05, 1.3120529639e-01, -1.8556939620e-01, 2.9365920892e-01, -9.1915488781e-02, -4.2094995876e-01, 2.0428342467e-01, 1.7394686270e-01, -4.7306070108e-05, -5.5245417297e-02, 8.0174676418e-02, -4.4812358812e-06, -5.5009670314e-05, 1.5573329497e-02, 2.6345535219e-02, 2.7347473322e-01, 1.3452874018e-01, -2.6552083051e-02, -3.4004765633e-04, 2.2096900185e-01, 9.3935515653e-07, -1.0448007845e-05, 5.2056362673e-07, 4.5842228920e-06, 3.9860469370e-01, 3.3628463474e-06, -2.1291504248e-06, 5.8799555941e-01, 4.2689440004e-01, 1.9140555781e-01, -1.8853159348e-01, 3.3254107845e-02, 1.5655542757e-01, 4.0798560932e-01, 2.1629607599e-02, 1.6452634164e-07, 2.3930890969e-06, 7.0857228736e-02, 2.4781060476e-01, -1.7675862067e-01, -1.5458719912e-04, 1.5834095155e-01, 3.4455896801e-01, 1.6935829932e-01, 2.6924841710e-01, -8.2805512369e-01, 5.0929939970e-01, 3.8690436695e-01, 2.1220254671e-01, -6.0987959753e-06, 7.0228944871e-02, 2.6424148910e-06, 1.5502642797e-02, 3.6028466430e-01, 1.0941150491e-01, 2.3142399026e-01, 3.2034413041e-01, 2.5161524646e-01, 2.8840344414e-06, 4.9456438188e-06, 2.2390150086e-01, 5.1931288885e-02, 1.1194833643e-01, 3.6147668502e-01, 2.3795810825e-01, -1.7920701003e-01, 1.2459069640e-01, -9.9294122406e-07, 7.4778025551e-02, 1.4017784128e-01, 4.8404518467e-01, 4.8519827907e-06, 2.3377260890e-01, 1.1146216963e-05, 1.0308204641e-07, -9.4236893910e-07, 2.3098981387e-01, -1.5366538292e-01, -2.7059698926e-01, -2.6877706289e-02, 1.5104147264e-01, 5.4276545227e-02, 1.4427219443e-05, 5.0270844599e-01, 1.1946769391e-01, 5.2722074806e-01, -3.2508527788e-01, 1.9877042275e-01, 1.4936059343e-01, 2.8380120273e-01, 5.7902590000e-01, 2.4716708046e-01, -2.0957255464e-06, 2.5109587961e-06, -1.5611229738e-01, 3.0413453893e-01, -5.5760405554e-07, 2.8431267215e-01, -4.3791169039e-02, 3.8319558441e-06, 1.9924935132e-01, 9.8612482074e-02, 1.3918156615e-05, 1.5065674934e-06, -3.6966399328e-06, 9.2354627455e-07, 2.0466048135e-01, 4.8831322488e-01, 1.5216697656e+00, -2.5785745963e-01, -1.4034277685e-05, 4.5762312762e-04, 2.5080595687e-01, -4.2241520806e-05, 2.1680775351e-01, 2.8221353962e-01, -3.1315951687e-01, 1.1848311909e-01, 6.9272285116e-01, -7.2392749937e-07, 1.5599749122e-01, -1.0485437368e-05, -5.2701599701e-08, 2.9263570258e-01, 3.7028388830e-06, 5.0769344858e-01, -4.0783181219e-06, 1.0690374715e-01, -1.2427479850e-02, -5.2685076849e-01, 1.3952157363e-01, -2.5578684884e-05, 2.0622447113e-01, 2.2016103653e-01, 6.4795267328e-01, -8.2115980578e-01, 7.7048844254e-06, -9.5494761405e-02]
params_cv=[-3.9839664653e-06, 1.9838223206e-09, 1.0946838457e-02, -3.4085156224e-06, 1.6314835918e-01, 6.0438869903e-01, 8.6104697665e-06, 2.4376328074e-07, 9.1033744962e-02, 1.9958342871e-01, -2.7181368421e-03, 2.6043020202e-01, -6.4039624473e-07, 2.2506004826e-01, 2.3020272226e-06, 1.6199874844e-01, -3.4832711218e-02, -3.5779751595e-01, -2.2633960365e-01, 6.6939111267e-03, -1.1208124939e-07, -1.3343552862e-01, 2.9088408381e-02, 3.8328351029e-07, 1.0923296927e-01, -3.6888660319e-01, 1.5894113249e-01, -3.7175638962e-02, -2.3993022226e-02, 2.7339632174e-01, -5.6709233204e-03, 4.0641128917e-01, 3.1534121216e-01, 5.7636138690e-06, 1.3058930667e-01, -1.2464773756e-02, 5.9122748485e-02, 8.4687517837e-01, 6.2274625216e-01, 2.4541358607e-01, 2.5079280001e-01, 5.7731482667e-01, 5.4877596474e-06, 8.9541313733e-06, 1.1016289750e-05, -9.2568406633e-02, 2.5071720851e-01, 5.1171372094e-02, 1.3232917803e-01, 1.4341562133e-01, 3.5009580564e-05, 2.2613805419e-05, 7.0894119140e-06, 6.6853769071e-02, 8.9425482687e-01, 1.0676430893e-01, 9.0781230724e-03, 9.6878899105e-02, -9.2332454437e-02, 5.5726206204e-03, 1.1257504682e-01, -1.6264607511e-01, 9.3247548767e-06, 1.7494318879e-01, 2.6585201420e-06, -7.3247680963e-03, -1.7741280813e-01, 1.8074044423e-01, 2.6316680775e-01, -1.9704148783e-02, 9.0458017712e-02, -2.2186703434e-03, -1.5732028984e-06, 5.7272265713e-02, -8.7356840970e-02, 2.5168172538e-01, -6.0242545076e-02, -2.8046818109e-01, 1.4957250380e-01, 1.0762229551e-01, -3.0571866087e-05, -2.1678746491e-03, 6.2718795640e-02, -3.3862454204e-06, 1.3435988965e-08, 2.5456993288e-02, 4.4159733580e-02, 2.6868288135e-01, 1.3985046775e-01, -1.2813846982e-06, -1.4532274449e-04, 2.2312844693e-01, -1.0003521563e-05, 1.8369464151e-06, -6.2441835402e-06, 5.8059233105e-06, 3.4306694680e-01, 1.9571741750e-06, -1.6249641317e-06, 4.8782648361e-01, 3.8902005408e-01, 1.8247810012e-01, -6.9171763143e-02, 7.7306577236e-06, 1.1145006954e-01, 3.4557526488e-01, 1.1126851456e-02, 1.2295772418e-06, 3.6998221670e-06, 3.4831259978e-03, 2.2009345714e-01, -1.2385749784e-01, 5.6852352022e-08, 1.3766083018e-01, 2.9210142259e-01, 6.1211591000e-02, 1.8853293503e-01, -5.6399453930e-01, 5.1503812312e-01, 3.9203831240e-01, 1.9301993894e-01, -8.2195603466e-07, 1.8387570582e-02, 3.2202882625e-06, 8.6846455951e-03, 3.1913801302e-01, 1.0308308963e-01, 1.5312663124e-01, 3.2268233978e-01, 2.3744985886e-01, 2.6609372286e-06, -2.2423368259e-06, 2.0032698445e-01, 6.6598408276e-02, 9.1092552477e-02, 3.0334169117e-01, 2.1155579947e-01, -4.7482788371e-02, 9.3264419434e-02, 9.1657803121e-07, 7.2700162397e-02, 1.3756784719e-01, 4.3835772366e-01, -7.2898794338e-07, 2.0386263416e-01, 1.6266408474e-05, -2.9958503242e-07, 2.3018909615e-06, 1.9820706680e-01, -1.0033895455e-01, -1.9705704762e-01, 2.6529540333e-06, 1.4247324396e-01, 3.8629317027e-02, 1.3130371543e-07, 4.6968413272e-01, 7.8639637354e-02, 3.7173816979e-01, -2.2910596563e-01, 1.8245098925e-01, 1.2422272765e-01, 2.7241481210e-01, 4.7982302635e-01, 3.3735372299e-01, 4.7259647218e-06, 1.1750138826e-05, -7.6487723285e-02, 1.8747333685e-01, 1.6482664828e-06, 2.1543864217e-01, -5.2693613814e-03, -2.0513783093e-04, 1.8001724761e-01, 1.5117943932e-01, -2.0989413106e-06, 3.3772105693e-06, 2.0590723574e-05, -3.9465573029e-07, 1.9125973928e-01, 4.4241872882e-01, 1.4684023254e+00, -1.9254676388e-01, -7.7806623028e-06, 1.5286906714e-04, 1.1384143643e-01, -9.5172773185e-06, 1.4889515736e-01, 2.4572951266e-01, -2.5882965512e-01, 1.0874059285e-01, 6.0413773457e-01, 3.9356695721e-06, 9.6726198653e-02, -1.1631766539e-06, -3.0178699804e-06, 1.9832110930e-01, 7.9523793803e-06, 3.5148688573e-01, -5.3305318907e-08, 5.1902680480e-04, -1.4897047212e-06, -3.6248722279e-01, 1.1345892956e-01, -1.6135025693e-06, 7.2823884734e-02, 9.0308122403e-02, 4.3055447058e-01, -5.8982133623e-01, 9.4843540773e-06, -4.8488103737e-06]


n=20
count=0
lookup=Hash.new
0.upto(n-2) { |i|
  (i+1).upto(n-1) { |j|
    lookup[[i,j]]=params_cv[count]
    count += 1
  }
}
n.times { |i|
  lookup[i]=params_cv[count+i]
}

lookup.keys.select { |i|
  i.class != Array
}.sort { |i,j| lookup[j].abs <=> lookup[i].abs }[0..19].each { |i|
  print "#{i} (#{qhash[qlist[i]]}: #{lookup[i]}\n"
}


lookup.keys.select { |i|
  i.class == Array
}.sort { |i,j| lookup[j].abs <=> lookup[i].abs }[0..19].each { |i|
  print "#{i} (#{qhash[qlist[i[0]]]} and #{qhash[qlist[i[1]]]}): #{lookup[i]}\n"
}

lookup.keys.select { |i|
  i.class == Array
}.sort { |i,j| lookup[j].abs <=> lookup[i].abs }.reverse[0..19].each { |i|
  print "#{i} (#{qhash[qlist[i[0]]]} and #{qhash[qlist[i[1]]]}): #{lookup[i]}\n"
}


