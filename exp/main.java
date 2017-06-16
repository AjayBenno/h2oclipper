  import java.io.*;
        import hex.genmodel.easy.RowData;
        import hex.genmodel.easy.EasyPredictModelWrapper;
        import hex.genmodel.easy.prediction.*;
        import hex.genmodel.MojoModel;
        public class main {
          public static void main(String[] args) throws Exception {
            EasyPredictModelWrapper model = new EasyPredictModelWrapper(MojoModel.load("GBM_model_python_1497584212592_1.zip"));

            RowData row = new RowData();
            row.put("AGE", "68");
            row.put("RACE", "2");
            row.put("DCAPS", "2");
            row.put("VOL", "0");
            row.put("GLEASON", "6");

            BinomialModelPrediction p = model.predictBinomial(row);
            System.out.println("Has penetrated the prostatic capsule (1=yes; 0=no): " + p.label);
            System.out.print("Class probabilities: ");
            for (int i = 0; i < p.classProbabilities.length; i++) {
              if (i > 0) {
                System.out.print(",");
              }
              System.out.print(p.classProbabilities[i]);
            }
            System.out.println("");
          }
        }
