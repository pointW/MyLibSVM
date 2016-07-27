package cn.pointw.mylibsvm.svm;

import cn.pointw.mylibsvm.libsvm.svm_scale;
import cn.pointw.mylibsvm.libsvm.svm_train;
import cn.pointw.mylibsvm.model.Item;
import cn.pointw.mylibsvm.model.Item2;
import cn.pointw.mylibsvm.util.*;
import com.google.gson.Gson;
import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import org.codehaus.jackson.map.ObjectMapper;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.security.spec.ECField;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

/**
 * Created by outen on 16/7/26.
 */
public class Main {
//    private static String dir = Main.class.getResource("../res").getPath().substring(1);
    private static String dir = "/Users/outen/develop/2016summerMachineLearning/MyLibSVM/res";

    public static void main(String[] args) throws Exception{
//        trainModel();
        predict();
    }

    public static void trainModel() throws Exception{
//        String fileName = "accdata2.csv";
        String fileName = "myData.csv";
//        List<Item> list = readFile(fileName);
        List<Item2> list = readFileToItem2(fileName);
        System.out.println(list.size());
        createTrainFile(list, "train");
        createScaleFile(new String[]{"-l","0","-u","1","-s",dir+"/range",dir+"/train"}, dir+"/scale");
        String[] cgr = cmdGridPy("python "+Util.getSvmPath()+"/libsvm/tools/grid.py"+" "+dir+"/scale",dir+"/grid");
        createModeFile(new String[]{"-s","0","-c",cgr[0],"-t","2","-g",cgr[1],"-e","0.1",dir+"/scale",dir+"/model"});
    }

    public static List<Item> readFile(String fileName) throws Exception{
        DateUtil.printNameDate(new Date(), "开始读取" + fileName + "文件");
        List<Item> list = new ArrayList<>();

        CSVFileUtil csv = new CSVFileUtil(dir + "/" + fileName);
        String temp = null;
        ArrayList<String> tempList = null;
        csv.readLine();
        while ((temp = csv.readLine()) != null){
            tempList = csv.fromCSVLinetoArray(temp);
            Item tempItem = new Item(tempList.get(1), tempList.get(2), Double.valueOf(tempList.get(3)),
                    Double.valueOf(tempList.get(4)), Double.valueOf(tempList.get(5)), Double.valueOf(tempList.get(6)),
                    Double.valueOf(tempList.get(7)), Double.valueOf(tempList.get(8)), Double.valueOf(tempList.get(9)),
                    Double.valueOf(tempList.get(10)), Double.valueOf(tempList.get(11)));
//            Item tempItem = new Item(tempList.get(0), tempList.get(1), Double.valueOf(tempList.get(2)),
//                    Double.valueOf(tempList.get(3)), Double.valueOf(tempList.get(4)), Double.valueOf(tempList.get(5)),
//                    Double.valueOf(tempList.get(6)), Double.valueOf(tempList.get(7)), Double.valueOf(tempList.get(8)),
//                    Double.valueOf(tempList.get(9)), Double.valueOf(tempList.get(10)));
            list.add(tempItem);
        }
        DateUtil.printNameDate(new Date(), "读取" + fileName + "完成");
        return list;
    }

    public static List<Item2> readFileToItem2(String fileName) throws Exception{
        DateUtil.printNameDate(new Date(), "开始读取"+fileName);
        List<Item2> list = new ArrayList<>();
        CSVFileUtil csv = new CSVFileUtil(dir + "/" + fileName);
        String temp = null;
        ArrayList<String> tempList = null;
        csv.readLine();
        while ((temp = csv.readLine()) != null){
            tempList = csv.fromCSVLinetoArray(temp);
            ObjectMapper mapper = new ObjectMapper();
            List<Double> stringList = new ArrayList<>();
            stringList = mapper.readValue(tempList.get(1), List.class);
            double[] acc = new double[128];
            for (int i = 0; i < acc.length; i++){
                acc[i] = stringList.get(i);
            }
            double[] fft = Features.fft(acc);
            Item2 item2 = new Item2(tempList.get(2), tempList.get(3), Features.energy(fft), Features.entropy(fft),
                    Features.iqr(acc), Features.mad(acc), Features.maximum(acc), Features.meanCrossingsRate(acc),
                    Features.mean(acc), Features.minimum(acc), Features.rms(acc), Features.spp(fft),
                    Features.standardDeviation(acc), Features.variance(acc));
            list.add(item2);
        }
        DateUtil.printNameDate(new Date(), "读取" + fileName + "完成");
        return list;
    }

//    public static void createTrainFile(List<Item> list, String trainFileName){
//        DateUtil.printNameDate(new Date(), "创建"+trainFileName);
//        StringBuffer stringBuffer = new StringBuffer();
//        Item tempItem = null;
//        for (int i = 0; i < list.size(); i++) {
//            tempItem = list.get(i);
//            stringBuffer.append(Constant.actMapToCode.get(tempItem.getAct()));
//            stringBuffer.append(" " + Constant.FUN_101_MINIMUM_CODE + ":" + tempItem.getT_min());
//            stringBuffer.append(" " + Constant.FUN_102_MAXIMUM_CODE + ":" + tempItem.getT_max());
//            stringBuffer.append(" " + Constant.FUN_103_VARIANCE_CODE + ":" + tempItem.getT_variance());
//            stringBuffer.append(" " + Constant.FUN_104_MEANCROSSINGSRATE_CODE + ":" + tempItem.getT_mcr());
//            stringBuffer.append(" " + Constant.FUN_105_STANDARDDEVIATION_CODE + ":" + tempItem.getT_sttdev());
//            stringBuffer.append(" " + Constant.FUN_106_MEAN_CODE + ":" + tempItem.getT_mean());
//            stringBuffer.append(" " + Constant.FUN_112_RMS_CODE + ":" + tempItem.getT_rms());
//            stringBuffer.append(" " + Constant.FUN_114_IQR_CODE + ":" + tempItem.getT_iqr());
//            stringBuffer.append(" " + Constant.FUN_115_MAD_CODE + ":" + tempItem.getT_mad());
//            stringBuffer.append(Util.getChangeRow());
//        }
//        Util.stringToFile(stringBuffer.toString(), dir+"/"+trainFileName, false);
//        DateUtil.printNameDate(new Date(), trainFileName+"创建完成");
//    }

    public static void createTrainFile(List<Item2> list, String trainFileName){
        DateUtil.printNameDate(new Date(), "创建"+trainFileName);
        StringBuffer stringBuffer = new StringBuffer();
        Item2 tempItem = null;
        for (int i = 0; i < list.size(); i++) {
            tempItem = list.get(i);
            stringBuffer.append(Constant.actMapToCode.get(tempItem.getAct()));
            stringBuffer.append(" " + Constant.FUN_101_MINIMUM_CODE + ":" + tempItem.getT_min());
            stringBuffer.append(" " + Constant.FUN_102_MAXIMUM_CODE + ":" + tempItem.getT_max());
            stringBuffer.append(" " + Constant.FUN_103_VARIANCE_CODE + ":" + tempItem.getT_variance());
            stringBuffer.append(" " + Constant.FUN_104_MEANCROSSINGSRATE_CODE + ":" + tempItem.getT_mcr());
            stringBuffer.append(" " + Constant.FUN_105_STANDARDDEVIATION_CODE + ":" + tempItem.getT_sttdev());
            stringBuffer.append(" " + Constant.FUN_106_MEAN_CODE + ":" + tempItem.getT_mean());
            stringBuffer.append(" " + Constant.FUN_112_RMS_CODE + ":" + tempItem.getT_rms());
            stringBuffer.append(" " + Constant.FUN_114_IQR_CODE + ":" + tempItem.getT_iqr());
            stringBuffer.append(" " + Constant.FUN_115_MAD_CODE + ":" + tempItem.getT_mad());

            stringBuffer.append(" " + Constant.FUN_201_SPP_CODE + ":" + tempItem.getT_spp());
            stringBuffer.append(" " + Constant.FUN_202_ENERGY_CODE + ":" + tempItem.getT_energy());
            stringBuffer.append(" " + Constant.FUN_203_ENTROPY_CODE + ":" +tempItem.getT_entropy());
            stringBuffer.append(Util.getChangeRow());
        }
        Util.stringToFile(stringBuffer.toString(), dir+"/"+trainFileName, false);
        DateUtil.printNameDate(new Date(), trainFileName+"创建完成");
    }

    public static void createScaleFile(String[] args, String scalePath){
        DateUtil.printNameDate(new Date(), "开始归一化");
        FileOutputStream fileOutputStream = null;
        PrintStream printStream = null;
        try{
            File file = new File(scalePath);
            file.createNewFile();
            fileOutputStream = new FileOutputStream(file);
            printStream = new PrintStream(fileOutputStream);
            PrintStream oldStream = System.out;
            System.setOut(printStream);
            svm_scale.main(args);
            System.setOut(oldStream);
        }
        catch (Exception e){
            e.printStackTrace();
        }
        finally {
            try {
                if(fileOutputStream!=null){
                    fileOutputStream.close();
                }
                if(printStream != null){
                    printStream.close();
                }
            }
            catch (IOException e) {
                e.printStackTrace();
            }
        }
        DateUtil.printNameDate(new Date(), "归一化结束");
    }

    public static String[] cmdGridPy(String str, String gridPath){
        DateUtil.printNameDate(new Date(), "开始计算cg的值");
        String grid = Util.exeCmd(str);
        System.out.println(str);
        Util.stringToFile(grid, gridPath, false);
        String gridEndLine = Util.readLastLine(new File(gridPath), null);
        gridEndLine = gridEndLine.substring(0, gridEndLine.indexOf("\n"));
        String[] cgr = gridEndLine.split(" ");
        DateUtil.printNameDate(new Date(), "cg值计算结束: c="+cgr[0]+" γ="+cgr[1]+" CV Rate="+cgr[2]+"%");
        return cgr;
    }

    public static void createModeFile(String[] args){
        DateUtil.printNameDate(new Date(), "开始计算model");
        try{
            svm_train.main(args);
        }
        catch (Exception e){
            e.printStackTrace();
        }
        DateUtil.printNameDate(new Date(), "计算model结束");
    }

    public static String[] readRule(String rulePath){
        String ruleStr = Util.readFileToString(rulePath);
        String[] ruleArr = ruleStr.split(Util.getChangeRow());
        return ruleArr;
    }

    public static void predict() throws Exception{
        String[] ruleArr = readRule(dir+"/range");
        createTrainFile(readFileToItem2("myData.csv"), "train2");
        int sum = ruleArr.length-2;
        String[] tempArr = null;
        tempArr = ruleArr[1].split(" ");
        double lower = Double.parseDouble(tempArr[0]);
        double upper = Double.parseDouble(tempArr[1]);

        String trainTest = Util.readFileToString(dir+"/train2");
        System.out.println(trainTest);
        String[] trainTestLineArr = trainTest.split(Util.getChangeRow());
        String[] trainTestItemArr = null;
        svm_node[] px = null;
        svm_node p = null;
        String[] tempNode = null;
        StringBuffer sb = new StringBuffer();
        System.out.println("预测结果 --- 真实结果 --- isTrue");
        int cw = 0;
        int zq = 0;
        for(int j = 0; j < trainTestLineArr.length; j++){
            trainTestItemArr = trainTestLineArr[j].split(" ");
            px = new svm_node[sum];
            for (int i = 0; i < 12; i++) {
                p = new svm_node();
                tempArr = ruleArr[i+2].split(" ");
                tempNode = trainTestItemArr[i+1].split(":");
                p.index = Integer.parseInt(tempNode[0]);
                p.value = Features.zeroOneLibSvm(lower, upper,
                        Double.parseDouble(tempNode[1]),
                        Double.parseDouble(tempArr[1]),
                        Double.parseDouble(tempArr[2]));
                px[i] = p;
            }
            svm_model model = svm.svm_load_model(dir+"/model");
            double code = svm.svm_predict(model, px);
            if (trainTestItemArr[0].equals(code+"")){
                System.out.println(code+"  "+trainTestItemArr[0]+" true");
                sb.append(code+"  "+trainTestItemArr[0]+" true");
                zq++;
            }
            else{
                System.err.println(code+"  "+trainTestItemArr[0]+" false");
                sb.append(code+"  "+trainTestItemArr[0]+" false");
                cw++;
            }
            sb.append(Util.getChangeRow());
        }
        System.out.println("预测结果: 正确: "+zq+" 错误: "+cw);
        Util.stringToFile(sb.toString(), dir+"/trainTestResult", false);
    }


}
