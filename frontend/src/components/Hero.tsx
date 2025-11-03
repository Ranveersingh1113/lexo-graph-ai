import { Button } from "@/components/ui/button";
import { ArrowRight, FileText, Sparkles } from "lucide-react";
import heroImage from "@/assets/hero-bg.jpg";

export const Hero = () => {
  const scrollToUpload = () => {
    document.getElementById("upload-section")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden">
      {/* Background with gradient overlay */}
      <div className="absolute inset-0 z-0">
        <div 
          className="absolute inset-0 bg-cover bg-center opacity-20"
          style={{ backgroundImage: `url(${heroImage})` }}
        />
        <div className="absolute inset-0 bg-gradient-to-b from-background via-background/95 to-background" />
      </div>

      {/* Animated gradient orbs */}
      <div className="absolute top-20 left-10 w-72 h-72 bg-primary/30 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-20 right-10 w-96 h-96 bg-accent/20 rounded-full blur-3xl animate-pulse delay-1000" />

      {/* Content */}
      <div className="relative z-10 container mx-auto px-6 text-center">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 mb-8 animate-fade-in">
          <Sparkles className="w-4 h-4 text-primary" />
          <span className="text-sm font-medium text-foreground">
            Advanced AI-Powered Document Analysis
          </span>
        </div>

        <h1 className="text-5xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-foreground via-primary to-accent bg-clip-text text-transparent leading-tight animate-fade-in-up">
          Intelligent Multilingual
          <br />
          Document Understanding
        </h1>

        <p className="text-xl md:text-2xl text-muted-foreground max-w-3xl mx-auto mb-12 leading-relaxed animate-fade-in-up delay-100">
          State-of-the-art layout analysis, OCR, and AI-powered content generation
          for complex multilingual documents
        </p>

        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center animate-fade-in-up delay-200">
          <Button 
            size="lg" 
            variant="hero"
            onClick={scrollToUpload}
            className="text-lg px-8 py-6 group"
          >
            Start Processing
            <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </Button>
          <Button 
            size="lg" 
            variant="outline"
            className="text-lg px-8 py-6"
          >
            <FileText className="w-5 h-5" />
            View Documentation
          </Button>
        </div>

        {/* Feature highlights */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-24 max-w-5xl mx-auto">
          {[
            { label: "mAP@0.5", value: "95%+", desc: "Layout Detection" },
            { label: "CER/WER", value: "<2%", desc: "OCR Accuracy" },
            { label: "Languages", value: "50+", desc: "Multilingual Support" },
          ].map((stat, idx) => (
            <div 
              key={idx} 
              className="p-6 rounded-2xl bg-card/50 backdrop-blur-sm border border-border/50 hover:border-primary/30 transition-all duration-300 hover:shadow-lg animate-fade-in-up"
              style={{ animationDelay: `${300 + idx * 100}ms` }}
            >
              <div className="text-3xl font-bold text-primary mb-2">{stat.value}</div>
              <div className="text-sm font-semibold text-foreground mb-1">{stat.label}</div>
              <div className="text-xs text-muted-foreground">{stat.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};
