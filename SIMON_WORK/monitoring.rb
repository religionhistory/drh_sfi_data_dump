require 'csv'

class String
  def hamming(other)
    n=self.length
    h=0
    n.times { |i|
      h += (self.split("")[i] != other.split("")[i] ? 1 : 0)
    }
    h
  end
end

# 21 questions
# 390 groups
file=File.new("../monitoring_questions.csv", 'r'); ans=CSV.parse(file.read); file.close

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

file=File.new("../SIMON_PROCESSED/processed_monitoring.csv", 'w')
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

file=File.new("monitoring.dat", 'w')
file.write("#{religion.keys.length}\n#{qlist.length}\n")
religion.keys.each { |i|
  file.write(qlist.collect { |j| religion[i][j] == "Yes" ? "1" : (religion[i][j] == "No" ? "0" : "X") }.join("")+" 1.0\n")
}
file.close

# `/Users/simon/Desktop/humanities-glass/MPF_CMU/mpf -c supreme_high_gods.dat 1.0`

params_neghalf=[2.6279926592e-08, -3.3565780288e-01, 1.6673802863e-01, -2.6206183589e-01, 5.9586266856e-02, -2.0509796539e-01, 1.7416216646e-07, 1.1656715288e-01, -1.1781781149e-07, 2.9544396473e-01, 3.0750756321e-01, 2.9147527388e-01, 1.8823179087e-01, 3.5515062586e-02, 5.0810886889e-01, -3.4927788591e-02, 2.8606477176e-01, 2.8744943801e-08, 3.6603687467e-02, 2.6035252330e-02, 4.4477021817e-01, 5.9207314810e-02, 1.2593850104e-01, 2.5191778880e-01, 1.4135827678e-01, 2.3526293615e-02, -2.1326725875e-01, 4.2417327995e-01, 1.2792830203e-01, 3.6795681764e-01, -2.2725434043e-01, -3.1505454473e-01, -2.6935226461e-01, 2.2927920643e-02, 1.1412230891e-01, 4.8778189425e-02, 3.1302911063e-01, -2.9689264863e-01, -1.1756882746e-01, 8.5983544805e-01, -1.6491714595e-08, 8.9877552342e-08, -4.4073910133e-01, -3.9808529292e-08, 2.4919171567e-02, -2.7147164288e-07, -1.4641888244e-01, -9.3576961147e-02, 5.4303860020e-03, -6.0064284892e-09, 9.9529779359e-02, -7.6365937451e-02, -1.6291567691e-01, -2.2664420832e-01, 6.7509316501e-02, -1.0258077042e-08, 1.7198757265e-01, -5.5090214358e-02, 1.9562450518e-01, 5.8556375787e-02, -3.1860877725e-03, -4.7228347578e-01, 2.8069634539e-08, 6.3881556473e-03, -4.9568423399e-01, 1.1387157910e-01, 2.8916147814e-01, 3.3958524831e-01, -4.2565835902e-03, 7.3220829705e-01, -1.3025423960e-01, 1.6009892581e-02, 7.2821746451e-01, -2.7752081593e-01, -1.5385518551e-01, -8.3319182492e-01, -6.8292272444e-09, 9.4213167709e-01, 5.9320346169e-01, 2.2694120627e-02, -1.5218103385e-08, -4.3030617537e-01, 2.6285918921e-01, 9.4984618633e-01, 3.0195851395e-07, 5.0557950434e-01, -2.5935281217e-02, 1.4784714947e-08, -3.7146838566e-01, 3.4997574062e-01, -2.1663660387e-07, 2.4547652976e-01, -5.5712304005e-09, 4.3344115045e-01, 2.7038746587e-09, 1.1222224626e+00, 1.2675280523e-13, 3.3870226435e-01, 9.3427940550e-04, 6.0983705692e-01, -3.4529883543e-01, 7.2309920901e-01, 5.2395627730e-01, 7.2717922298e-02, 6.3797831662e-02, -4.7796831727e-07, 3.1561456883e-03, 5.7218504325e-08, -9.4808288434e-08, -7.6079754419e-07, -3.3979287391e-02, -2.3120634739e-08, 1.2245445136e-01, 4.4930378364e-08, 3.9623950982e-01, -2.5199694379e-08, -2.2304649082e-01, -3.4071305010e-01, 4.6159983112e-01, 6.2467817866e-08]
params_neghalf=params_negone=[-7.1500805518e-03, -5.5382573836e-01, 2.8567705832e-01, -3.7188630364e-01, 3.0197535070e-01, -4.3370715695e-01, 1.5610160584e-01, 4.3591640248e-05, -2.6997310105e-02, 5.0929244043e-01, 3.0101429965e-01, 3.1349850403e-01, 1.4405132350e-01, 8.6449558375e-02, 7.2776246688e-01, -2.6625441383e-02, 3.7390344936e-01, -7.9296615009e-02, 1.3065400503e-01, 3.1155659850e-01, 5.7753070580e-01, 3.0253939397e-02, 8.3116268938e-03, 3.2753045829e-01, 2.1787816804e-01, 6.5549896878e-02, -4.4903128635e-01, 5.9545675455e-01, 5.4621727464e-02, 5.6738241069e-01, -5.5800296002e-01, -5.1237004929e-01, -9.9049549537e-01, 5.7214188395e-01, 9.1455711387e-01, -1.4258425302e-05, 4.4874853923e-01, -6.6806425956e-01, -2.0326969922e-01, 8.2428588271e-01, -1.7595151923e-01, 2.8167694583e-01, -9.8472463115e-01, 6.6250575338e-05, 7.2205536612e-01, -1.8130124341e-05, -3.0267611984e-01, -2.1493441798e-01, 1.1896704874e-01, 5.4053733529e-01, 1.5885895316e-01, -2.4310917722e-01, -5.2466532497e-01, -4.4204237267e-01, 6.9958615494e-01, 3.6457905559e-01, 1.6725588601e-01, -2.0683814508e-01, 2.4915697958e-01, 7.5618479806e-01, 1.7213685500e-01, -1.4832523648e-01, 1.2813103365e-01, 6.0899906638e-01, -1.0385930012e+00, 3.5299856678e-01, 2.4650740378e-01, 5.8616266833e-01, -1.1531329160e-01, 1.1878510539e+00, -6.8909487187e-01, 1.4924083963e-01, 1.4142954808e+00, -5.4218840076e-01, -1.3172652771e-01, -1.2024228668e+00, -3.2100282521e-05, 7.0196043784e-01, 3.4197160608e-02, 6.5592727201e-01, -9.8216012993e-02, -1.1657933832e+00, 6.5830464367e-01, 2.0404487863e+00, 5.8960388311e-01, 8.0265824540e-01, -2.9476426971e-01, 4.4089359184e-01, -9.1348372913e-01, 5.6089073525e-01, -2.2317629873e-01, 3.9474772070e-01, 2.8489990391e-07, 4.6848094180e-01, -2.9422308913e-01, 1.4145049301e+00, 3.9308284056e-01, 6.0790235494e-01, -6.9963227200e-02, 6.2186481587e-01, -5.1016010433e-01, 9.1135192278e-01, 6.8125115301e-01, 5.6531086927e-01, 1.4981447970e-01, -6.2072997540e-13, 9.6863809861e-02, -2.9492140261e-02, -8.9404246003e-01, -1.0489989967e+00, -1.0436265724e+00, 2.8455472708e-06, 8.0417317526e-01, 4.7991953306e-06, 2.9678875647e+00, 3.4570819347e-05, -1.4591425865e-01, -1.0831817852e+00, 8.2757662558e-01, 5.1651983998e-01]

`../../humanities-glass/MPF_CMU/mpf -z supreme_high_gods.dat_params.dat_LAMBDA-1 15`

file=File.new("supreme_high_gods.dat_params.dat_probs.dat", 'r')
prob=Hash.new
file.each_line { |i|
  prob[i.split(" ")[0]]=i.split(" ")[-1].to_f
}; file.close

corr=Hash.new
n=15
count=0
lookup=Hash.new
0.upto(n-2) { |i|
  (i+1).upto(n-1) { |j|
    lookup[[i,j]]=params_neghalf[count]
    one_one=prob.keys.select { |unit| unit[i] == "1" and unit[j] == "1" }.collect { |i| prob[i] }.sum
    one_zero=prob.keys.select { |unit| unit[i] == "1" and unit[j] == "0" }.collect { |i| prob[i] }.sum
    zero_one=prob.keys.select { |unit| unit[i] == "0" and unit[j] == "1" }.collect { |i| prob[i] }.sum
    zero_zero=prob.keys.select { |unit| unit[i] == "0" and unit[j] == "0" }.collect { |i| prob[i] }.sum
    corr[[i,j]]=one_one-one_zero-zero_one+zero_zero
    count += 1
  }
}
n.times { |i|
  lookup[i]=params_neghalf[count+i]
  corr[i]=prob.keys.select { |unit| unit[i] == "1" }.collect { |i| prob[i] }.sum - prob.keys.select { |unit| unit[i] == "0" }.collect { |i| prob[i] }.sum
}

load '../../ENT/ent.rb'
mi=Array.new(n) { |loc|
  h=Array.new(1 << (n-1)) { |unit|
    count=0
    lo=Array.new(n) { |i|
      if i == loc then
        0
      else
        bin=unit.to_s(2)
        bin="0"*(n-bin.length)+bin
        ans=bin[count]
        count += 1
        ans
      end
    }
    p=prob[lo.join("")]
    lo[loc]=1
    p += prob[lo.join("")]
  }.ent
  hlo=Array.new(1 << (n-1)) { |unit|
    count=0
    lo=Array.new(n) { |i|
      if i == loc then
        0
      else
        bin=unit.to_s(2)
        bin="0"*(n-bin.length)+bin
        ans=bin[count]
        count += 1
        ans
      end
    }
    prob[lo.join("")]
  }.ent*prob.keys.select { |i| i[loc] == "0" }.collect { |i| prob[i] }.sum
  hhi=Array.new(1 << (n-1)) { |unit|
    count=0
    lo=Array.new(n) { |i|
      if i == loc then
        1
      else
        bin=unit.to_s(2)
        bin="0"*(n-bin.length)+bin
        ans=bin[count]
        count += 1
        ans
      end
    }
    prob[lo.join("")]
  }.ent*prob.keys.select { |i| i[loc] == "1" }.collect { |i| prob[i] }.sum
  mi=h-(hlo+hhi)
  [loc, mi]
}
mi.sort { |i,j| j[1] <=> i[1] }.each { |i|
  print "#{i[0]} (#{qhash[qlist[i[0]]][0..-2]}): #{i[1]}\n"
}

lookup.keys.select { |i|
  i.class == Array
}.collect { |i|
  [i[0], i[1], corr[i], lookup[i]]
}

lookup.keys.select { |i|
  i.class != Array
}.sort { |i,j| lookup[j].abs <=> lookup[i].abs }[0..19].each { |i|
  print "#{i} (#{qhash[qlist[i]]}: #{lookup[i]} (#{corr[i]})\n"
}

lookup.keys.select { |i|
  i.class == Array
}.sort { |i,j| lookup[j].abs <=> lookup[i].abs }[0..19].each { |i|
  print "#{i} (#{qhash[qlist[i[0]]]} and #{qhash[qlist[i[1]]]}): #{lookup[i]} #{corr[i]}\n"
}

file=File.new("edges.csv", 'w')
file.write("Source,Target,Weight,Sign\n")
lookup.keys.select { |i|
  i.class == Array and lookup[i].abs > 1e-3
}.each { |i|
  file.write("#{i[0]},#{i[1]},#{lookup[i].abs},#{lookup[i] < 0 ? -1 : 1}\n")
}
file.close
file=File.new("nodes.csv", 'w')
file.write("Id,Label\n")
n.times { |i|
  file.write("#{i},#{i}\n")
}; file.close

lookup.keys.select { |i|
  i.class == Array
}.sort { |i,j| corr[j].abs <=> corr[i].abs }[0..19].each { |i|
  print "#{i} (#{qhash[qlist[i[0]]]} and #{qhash[qlist[i[1]]]}): #{lookup[i]} #{corr[i]}\n"
}

lookup.keys.select { |i|
  i.class == Array
}.sort { |i,j| lookup[j].abs <=> lookup[i].abs }.reverse[0..19].each { |i|
  print "#{i} (#{qhash[qlist[i[0]]]} and #{qhash[qlist[i[1]]]}): #{lookup[i]}\n"
}

