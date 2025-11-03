import { Card } from "@/components/ui/card";
import { 
  FileSearch, 
  Languages, 
  Zap, 
  LineChart, 
  Shield, 
  Sparkles 
} from "lucide-react";

const features = [
  {
    icon: FileSearch,
    title: "Advanced Layout Analysis",
    description: "State-of-the-art document structure detection with PP-DocLayout and LayoutXLM for precise element classification.",
    gradient: "from-blue-500 to-cyan-500",
  },
  {
    icon: Languages,
    title: "Multilingual OCR",
    description: "Support for 50+ languages including complex scripts like Urdu, Arabic, Persian, and mixed-language documents.",
    gradient: "from-purple-500 to-pink-500",
  },
  {
    icon: Sparkles,
    title: "AI-Powered Generation",
    description: "Automatic generation of natural language descriptions for charts, tables, and figures using advanced LLMs.",
    gradient: "from-violet-500 to-purple-500",
  },
  {
    icon: Zap,
    title: "High Performance",
    description: "Optimized inference pipeline leveraging GPU acceleration for real-time processing of complex documents.",
    gradient: "from-orange-500 to-red-500",
  },
  {
    icon: LineChart,
    title: "Superior Metrics",
    description: "Achieving 95%+ mAP for layout detection and <2% CER/WER for text recognition across all supported languages.",
    gradient: "from-green-500 to-emerald-500",
  },
  {
    icon: Shield,
    title: "Research-Grade Quality",
    description: "Built on peer-reviewed methodologies and validated against benchmark datasets for academic applications.",
    gradient: "from-indigo-500 to-blue-500",
  },
];

export const Features = () => {
  return (
    <section className="py-24 px-6 bg-muted/30">
      <div className="container mx-auto max-w-7xl">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            Cutting-Edge Technology
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Built on state-of-the-art models and research-backed methodologies
            for unparalleled document understanding
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <Card
              key={index}
              className="p-8 hover:shadow-xl transition-all duration-300 hover:-translate-y-1 border-border/50 bg-card/80 backdrop-blur-sm group"
            >
              <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                <feature.icon className="w-7 h-7 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
              <p className="text-muted-foreground leading-relaxed">
                {feature.description}
              </p>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};
