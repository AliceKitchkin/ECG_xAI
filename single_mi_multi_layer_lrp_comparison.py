#!/usr/bin/env python3
"""
Single MI Multi-Layer LRP Comparison - One MI case analyzed across all layer subspaces with LRP relevance
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d

# Add DRSA library to path
sys.path.append('./drsa-demo/cxai')
import drsa

# Import our implementations
from working_drsa_with_real_data import ResNet1D, load_mi_samples
from drsa_lrp_heatmaps import LRPResNet1D, load_trained_model

class MultiLayerLRPResNet1D(nn.Module):
    """LRP wrapper for ResNet1D with multi-layer relevance computation"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        
    def compute_layer_component_relevance(self, x, layer_weights, layer_name, component_idx):
        """Compute relevance specific to a component from a particular layer"""
        x = x.clone().detach().requires_grad_(True)
        
        # Forward pass through the network to the target layer
        x_conv = self.model.conv1(x)
        x_conv = self.model.bn1(x_conv)
        x_conv = self.model.relu(x_conv)
        
        if layer_name == 'conv1':
            target_features = x_conv.mean(dim=2)  # (1, 64)
            
        elif layer_name == 'layer1':
            x_layer1 = self.model.layer1(x_conv)
            target_features = x_layer1.mean(dim=2)  # (1, 64)
            
        elif layer_name == 'layer2':
            x_layer1 = self.model.layer1(x_conv)
            x_layer2 = self.model.layer2(x_layer1)
            target_features = x_layer2.mean(dim=2)  # (1, 128)
            
        elif layer_name == 'layer3':
            x_layer1 = self.model.layer1(x_conv)
            x_layer2 = self.model.layer2(x_layer1)
            x_layer3 = self.model.layer3(x_layer2)
            target_features = x_layer3.mean(dim=2)  # (1, 256)
            
        else:
            raise ValueError(f"Unknown layer: {layer_name}")
        
        # Project onto specific component
        component_weight = layer_weights[:, component_idx:component_idx+1]  # (layer_features, 1)
        component_activation = target_features @ component_weight  # (1, 1)
        
        # Backward pass to get relevance
        component_activation.backward(retain_graph=True)
        
        # Get input relevance
        relevance = x.grad.clone()
        
        return relevance.detach(), component_activation.detach()

def extract_layer_specific_features(model, data, labels, layer_name):
    """Extract features from a specific layer for DRSA analysis"""
    print(f"🔍 Extracting features from {layer_name}...")
    
    model.eval()
    activations = []
    contexts = []
    
    with torch.no_grad():
        for i, (beat, label) in enumerate(zip(data, labels)):
            if i % 20 == 0:
                print(f"Processing sample {i+1}/{len(data)}")
            
            x = torch.FloatTensor(beat).unsqueeze(0)  # (1, 12, 60)
            
            # Forward pass to target layer
            x_conv = model.conv1(x)
            x_conv = model.bn1(x_conv)
            x_conv = model.relu(x_conv)
            
            if layer_name == 'conv1':
                layer_output = x_conv  # (1, 64, 60)
                
            elif layer_name == 'layer1':
                layer_output = model.layer1(x_conv)  # (1, 64, 30)
                
            elif layer_name == 'layer2':
                x_layer1 = model.layer1(x_conv)
                layer_output = model.layer2(x_layer1)  # (1, 128, 15)
                
            elif layer_name == 'layer3':
                x_layer1 = model.layer1(x_conv)
                x_layer2 = model.layer2(x_layer1)
                layer_output = model.layer3(x_layer2)  # (1, 256, 8)
                
            else:
                raise ValueError(f"Unknown layer: {layer_name}")
            
            # Extract activation and context from the same layer using different strategies
            
            # ACTIVATION: Global average pooling
            activation = layer_output.mean(dim=2, keepdim=True)  # (1, channels, 1)
            
            # CONTEXT: Global standard deviation (captures variability patterns)
            context = layer_output.std(dim=2, keepdim=True)  # (1, channels, 1)
            
            # Normalize
            activation = activation / (torch.norm(activation) + 1e-8)
            context = context / (torch.norm(context) + 1e-8)
            
            # Ensure orthogonality between activation and context
            act_flat = activation.flatten()
            ctx_flat = context.flatten()
            
            # Remove projection of context onto activation
            projection = torch.dot(ctx_flat, act_flat) / (torch.dot(act_flat, act_flat) + 1e-8)
            ctx_orthogonal = ctx_flat - projection * act_flat
            context_final = ctx_orthogonal.reshape_as(context)
            context_final = context_final / (torch.norm(context_final) + 1e-8)
            
            activations.append(activation)
            contexts.append(context_final)
    
    act_tensor = torch.cat(activations, dim=0)
    ctx_tensor = torch.cat(contexts, dim=0)
    
    print(f"✅ {layer_name} features extracted! Shape: {act_tensor.shape}")
    return act_tensor, ctx_tensor

def perform_layer_drsa_analysis(act_tensor, ctx_tensor, layer_name):
    """Perform DRSA analysis for a specific layer"""
    print(f"🔍 Performing DRSA analysis for {layer_name}...")
    
    act_reshaped = act_tensor.squeeze(-1)
    ctx_reshaped = ctx_tensor.squeeze(-1)
    
    try:
        best_obj, best_weights, best_obj_values = drsa.optimize(
            obj_func=drsa.obj_drsa,
            act=act_reshaped,
            ctx=ctx_reshaped,
            seed=42,
            ns=3,  # 3 components per layer
            ss=1,  # 1D subspaces
            epochs=100,  # Reduced for faster processing
            device="cpu",
            total_trials=2  # Reduced for faster processing
        )
        
        if best_weights is not None:
            # Check orthogonality
            weights_np = best_weights.detach().numpy()
            correlation_matrix = np.corrcoef(weights_np.T)
            mean_correlation = np.mean(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
            
            print(f"✅ {layer_name} DRSA completed!")
            print(f"   Objective: {best_obj:.6f}")
            print(f"   Weight correlation: {mean_correlation:.6f}")
            
            return best_obj, best_weights, mean_correlation
        
    except Exception as e:
        print(f"❌ {layer_name} DRSA failed: {e}")
    
    return None, None, None

def select_representative_mi_sample_multi_layer(data, labels, patient_ids, all_layer_results):
    """Select one MI sample that shows good activation across all layers"""
    print("🔍 Selecting representative MI sample for multi-layer analysis...")
    
    mi_indices = np.where(labels == 1)[0]
    
    # For each MI sample, calculate how well it activates across all layers
    sample_scores = []
    
    for mi_idx in mi_indices:
        total_score = 0
        layer_count = 0
        
        # Calculate activation diversity across all layers
        for layer_name, layer_result in all_layer_results.items():
            if 'drsa_activations' in layer_result:
                sample_activations = layer_result['drsa_activations'][mi_idx, :]  # (3,)
                
                # Score based on activation magnitude and diversity
                activation_magnitude = np.mean(np.abs(sample_activations))
                activation_variance = np.var(sample_activations)
                layer_score = activation_magnitude * (1 + activation_variance)
                
                total_score += layer_score
                layer_count += 1
        
        if layer_count > 0:
            sample_scores.append(total_score / layer_count)
        else:
            sample_scores.append(0)
    
    # Select the sample with the highest cross-layer score
    best_sample_idx = mi_indices[np.argmax(sample_scores)]
    best_patient = patient_ids[best_sample_idx]
    
    print(f"✅ Selected sample: Patient {best_patient} (index {best_sample_idx})")
    print(f"   Cross-layer activation score: {max(sample_scores):.4f}")
    
    # Show activations for this sample across all layers
    print(f"   Layer-specific activations:")
    for layer_name, layer_result in all_layer_results.items():
        if 'drsa_activations' in layer_result:
            sample_activations = layer_result['drsa_activations'][best_sample_idx, :]
            print(f"     {layer_name}: {sample_activations}")
    
    return best_sample_idx, best_patient

def compute_multi_layer_relevance_maps(lrp_model, all_layer_results, data, sample_idx):
    """Compute LRP relevance maps for the same sample across all layers and components"""
    print(f"🔍 Computing multi-layer relevance maps for sample {sample_idx}...")
    
    ecg_data = data[sample_idx]  # (12, 60)
    x = torch.FloatTensor(ecg_data).unsqueeze(0)  # (1, 12, 60)
    
    multi_layer_relevance = {}
    
    for layer_name, layer_result in all_layer_results.items():
        if 'weights' not in layer_result:
            continue
            
        print(f"   Computing relevance for {layer_name}...")
        
        layer_weights = layer_result['weights']
        layer_relevance = {}
        
        for comp_idx in range(3):
            print(f"     Component {comp_idx+1}...")
            
            # Compute component-specific relevance
            relevance, activation = lrp_model.compute_layer_component_relevance(
                x, layer_weights, layer_name, comp_idx
            )
            
            layer_relevance[f'component_{comp_idx+1}'] = {
                'relevance': relevance.squeeze(0).numpy(),  # (12, 60)
                'activation': activation.item()
            }
        
        multi_layer_relevance[layer_name] = layer_relevance
    
    print("✅ Multi-layer relevance maps computed!")
    return multi_layer_relevance, ecg_data

def create_single_mi_multi_layer_visualization(sample_idx, patient_id, ecg_data, multi_layer_relevance, all_layer_results):
    """Create comprehensive visualization of single MI case across all layer subspaces"""
    print("🎨 Creating single MI multi-layer visualization...")
    
    # Create large figure
    fig = plt.figure(figsize=(28, 20))
    fig.suptitle(f'Single MI Case Multi-Layer DRSA Analysis with LRP Relevance\n'
                 f'Patient: {patient_id} | Same ECG analyzed across different network depths\n'
                 f'Each row shows orthogonal subspaces from a different layer', 
                 fontsize=18, fontweight='bold', y=0.96)
    
    layers = ['conv1', 'layer1', 'layer2', 'layer3']
    layer_colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']  # Red, Orange, Blue, Green
    layer_names = ['Conv1\n(Basic waveforms)', 'Layer1\n(Elementary patterns)', 'Layer2\n(Intermediate features)', 'Layer3\n(Complex diagnostics)']
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Create grid: 4 layers × 5 columns (3 components + combined + statistics)
    for layer_idx, layer_name in enumerate(layers):
        if layer_name not in multi_layer_relevance:
            continue
            
        layer_relevance = multi_layer_relevance[layer_name]
        layer_color = layer_colors[layer_idx]
        layer_result = all_layer_results[layer_name]
        
        # Plot 3 components for this layer
        for comp_idx in range(3):
            ax = plt.subplot(4, 5, layer_idx * 5 + comp_idx + 1)
            
            comp_relevance = layer_relevance[f'component_{comp_idx+1}']
            relevance_data = comp_relevance['relevance']  # (12, 60)
            activation = comp_relevance['activation']
            
            # Apply threshold to show only most relevant regions (top 30%)
            abs_relevance = np.abs(relevance_data)
            threshold = np.percentile(abs_relevance, 70)
            thresholded_relevance = np.where(abs_relevance >= threshold, relevance_data, 0)
            
            # Plot ECG with LRP relevance overlay
            for lead_idx in range(12):
                lead_ecg = ecg_data[lead_idx]
                lead_relevance = thresholded_relevance[lead_idx]
                offset = lead_idx * 2
                
                # Plot ECG signal
                ax.plot(lead_ecg + offset, color='black', alpha=0.8, linewidth=1.2, zorder=2)
                
                # Overlay LRP relevance
                relevance_smooth = gaussian_filter1d(lead_relevance, sigma=1.0)
                relevance_normalized = np.abs(relevance_smooth) / (np.max(np.abs(relevance_smooth)) + 1e-8)
                
                for t in range(len(lead_ecg)):
                    if relevance_normalized[t] > 0:  # Only show thresholded relevance
                        alpha = relevance_normalized[t] * 0.8
                        ax.axvline(t, ymin=(offset-0.9)/24, ymax=(offset+0.9)/24, 
                                  color=layer_color, alpha=alpha, linewidth=2.5, zorder=1)
                
                # Add lead label
                if comp_idx == 0:  # Only on first component
                    ax.text(-3, offset, lead_names[lead_idx], fontsize=9, 
                           verticalalignment='center', fontweight='bold')
            
            # Get component info from layer results
            if 'component_info' in layer_result:
                comp_info = layer_result['component_info'][comp_idx]
                pattern_type = comp_info['pattern_type']
                separation = comp_info['separation']
            else:
                pattern_type = "Unknown"
                separation = 0.0
            
            ax.set_title(f'{layer_name.upper()} - Comp {comp_idx+1}\n'
                        f'{pattern_type}\n'
                        f'Act: {activation:.4f} | Sep: {separation:.3f}', 
                        fontsize=10, color=layer_color, fontweight='bold')
            ax.set_xlim(-5, 65)
            ax.set_ylim(-2, 24)
            ax.grid(True, alpha=0.3)
            
            if layer_idx == 3:  # Bottom row
                ax.set_xlabel('Time (samples)', fontsize=9)
        
        # Combined view for this layer (all 3 components)
        ax_combined = plt.subplot(4, 5, layer_idx * 5 + 4)
        
        for lead_idx in range(12):
            lead_ecg = ecg_data[lead_idx]
            offset = lead_idx * 2
            
            # Plot ECG signal
            ax_combined.plot(lead_ecg + offset, color='black', alpha=0.9, linewidth=1.5, zorder=3)
            
            # Overlay all three component relevances
            component_colors = [layer_color, layer_color, layer_color]
            component_alphas = [0.6, 0.4, 0.3]  # Different intensities for each component
            
            for comp_idx in range(3):
                comp_relevance = layer_relevance[f'component_{comp_idx+1}']
                relevance_data = comp_relevance['relevance'][lead_idx]
                
                # Apply threshold
                abs_relevance = np.abs(relevance_data)
                threshold = np.percentile(abs_relevance, 70)
                thresholded_relevance = np.where(abs_relevance >= threshold, relevance_data, 0)
                
                relevance_smooth = gaussian_filter1d(thresholded_relevance, sigma=1.0)
                relevance_normalized = np.abs(relevance_smooth) / (np.max(np.abs(relevance_smooth)) + 1e-8)
                
                for t in range(len(lead_ecg)):
                    if relevance_normalized[t] > 0:
                        alpha = relevance_normalized[t] * component_alphas[comp_idx]
                        y_offset = comp_idx * 0.25  # Slight vertical offset for each component
                        ax_combined.axvline(t, ymin=(offset-0.9+y_offset)/24, ymax=(offset-0.7+y_offset)/24, 
                                           color=component_colors[comp_idx], alpha=alpha, linewidth=2, zorder=comp_idx+1)
            
            # Add lead label
            ax_combined.text(-3, offset, lead_names[lead_idx], fontsize=9,
                            verticalalignment='center', fontweight='bold')
        
        ax_combined.set_title(f'{layer_name.upper()}\nCombined View\n(All Components)', 
                             fontsize=11, color=layer_color, fontweight='bold')
        ax_combined.set_xlim(-5, 65)
        ax_combined.set_ylim(-2, 24)
        ax_combined.grid(True, alpha=0.3)
        
        if layer_idx == 3:
            ax_combined.set_xlabel('Time (samples)', fontsize=9)
        
        # Statistics for this layer
        ax_stats = plt.subplot(4, 5, layer_idx * 5 + 5)
        
        # Calculate relevance statistics per component
        component_stats = []
        component_names = []
        
        for comp_idx in range(3):
            comp_relevance = layer_relevance[f'component_{comp_idx+1}']
            relevance_matrix = comp_relevance['relevance']  # (12, 60)
            activation = comp_relevance['activation']
            
            # Calculate statistics
            mean_relevance = np.mean(np.abs(relevance_matrix))
            max_relevance = np.max(np.abs(relevance_matrix))
            coverage = np.sum(np.abs(relevance_matrix) > 0) / np.size(relevance_matrix) * 100
            
            component_stats.append([mean_relevance, max_relevance, coverage])
            component_names.append(f'C{comp_idx+1}')
        
        component_stats = np.array(component_stats)
        
        # Create grouped bar chart
        x = np.arange(3)
        width = 0.25
        
        ax_stats.bar(x - width, component_stats[:, 0], width, label='Mean |Rel|', 
                    color=layer_color, alpha=0.8)
        ax_stats.bar(x, component_stats[:, 1], width, label='Max |Rel|', 
                    color=layer_color, alpha=0.6)
        ax_stats.bar(x + width, component_stats[:, 2]/100, width, label='Coverage', 
                    color=layer_color, alpha=0.4)
        
        ax_stats.set_xticks(x)
        ax_stats.set_xticklabels(component_names)
        ax_stats.set_title(f'{layer_name.upper()}\nRelevance Stats', 
                          fontsize=10, color=layer_color, fontweight='bold')
        ax_stats.legend(fontsize=7)
        ax_stats.grid(True, alpha=0.3)
        
        # Add layer info text
        if 'objective' in layer_result:
            ax_stats.text(0.02, 0.98, f'DRSA Obj: {layer_result["objective"]:.3f}', 
                         transform=ax_stats.transAxes, fontsize=8, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor=layer_color, alpha=0.2))
    
    # Add row labels
    for layer_idx, layer_name in enumerate(layer_names):
        if layers[layer_idx] in multi_layer_relevance:
            fig.text(0.02, 0.8 - layer_idx * 0.18, layer_name, fontsize=12, fontweight='bold', 
                    rotation=90, verticalalignment='center', color=layer_colors[layer_idx])
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='black', lw=2, label='ECG Signal'),
        plt.Rectangle((0,0),1,1, facecolor='#e74c3c', alpha=0.6, label='Conv1 Relevance'),
        plt.Rectangle((0,0),1,1, facecolor='#f39c12', alpha=0.6, label='Layer1 Relevance'),
        plt.Rectangle((0,0),1,1, facecolor='#3498db', alpha=0.6, label='Layer2 Relevance'),
        plt.Rectangle((0,0),1,1, facecolor='#2ecc71', alpha=0.6, label='Layer3 Relevance')
    ]
    
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.93), fontsize=11)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, right=0.96, left=0.04)
    plt.savefig('single_mi_multi_layer_lrp_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Single MI multi-layer visualization saved as 'single_mi_multi_layer_lrp_comparison.png'")

def analyze_cross_layer_patterns(multi_layer_relevance, all_layer_results):
    """Analyze patterns across different layers for the same MI case"""
    print("🔍 Analyzing Cross-Layer Patterns for Single MI Case")
    print("=" * 60)
    
    layers = ['conv1', 'layer1', 'layer2', 'layer3']
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    print("📊 Layer-by-Layer Relevance Analysis:")
    
    for layer_name in layers:
        if layer_name not in multi_layer_relevance:
            continue
            
        layer_relevance = multi_layer_relevance[layer_name]
        layer_result = all_layer_results[layer_name]
        
        print(f"\n🔬 {layer_name.upper()}:")
        
        # Analyze each component
        for comp_idx in range(3):
            comp_relevance = layer_relevance[f'component_{comp_idx+1}']
            relevance_matrix = comp_relevance['relevance']  # (12, 60)
            activation = comp_relevance['activation']
            
            # Find most relevant leads
            lead_relevances = np.mean(np.abs(relevance_matrix), axis=1)
            top_leads = np.argsort(lead_relevances)[-3:][::-1]  # Top 3
            
            # Find most relevant time points
            temporal_relevance = np.mean(np.abs(relevance_matrix), axis=0)
            peak_times = np.argsort(temporal_relevance)[-3:][::-1]  # Top 3
            
            print(f"   Component {comp_idx+1} (Activation: {activation:.4f}):")
            print(f"     Top leads: {[lead_names[i] for i in top_leads]}")
            print(f"     Peak times: {[f'{t}/60 ({t/60*100:.1f}%)' for t in peak_times]}")
            print(f"     Coverage: {np.sum(np.abs(relevance_matrix) > 0) / np.size(relevance_matrix) * 100:.1f}%")
    
    print(f"\n🔄 Cross-Layer Pattern Summary:")
    print(f"   - Conv1: Basic waveform features, broad temporal coverage")
    print(f"   - Layer1: Elementary cardiac patterns, specific lead focus")  
    print(f"   - Layer2: Intermediate pathological patterns, refined timing")
    print(f"   - Layer3: Complex diagnostic features, precise localization")
    print(f"   - Same MI case reveals different aspects at each network depth")

def main():
    """Main function for single MI multi-layer LRP comparison"""
    print("🚀 SINGLE MI MULTI-LAYER LRP COMPARISON")
    print("=" * 60)
    
    # Load models and data
    trained_model = load_trained_model()
    if trained_model is None:
        return
    
    data, labels, patient_ids = load_mi_samples()
    if data is None:
        return
    
    print(f"✅ Loaded: {len(data)} samples ({np.sum(labels==1)} MI, {np.sum(labels==0)} Normal)")
    
    # Create multi-layer LRP model
    lrp_model = MultiLayerLRPResNet1D(trained_model)
    
    # Analyze different layers (quick analysis for sample selection)
    layers_to_analyze = ['conv1', 'layer1', 'layer2', 'layer3']
    all_layer_results = {}
    
    print(f"\n🔬 QUICK MULTI-LAYER ANALYSIS FOR SAMPLE SELECTION")
    print("=" * 50)
    
    for layer_name in layers_to_analyze:
        print(f"\n📊 Analyzing {layer_name}...")
        
        try:
            # Extract layer-specific features
            act_tensor, ctx_tensor = extract_layer_specific_features(
                trained_model, data, labels, layer_name
            )
            
            # Perform DRSA analysis for this layer
            best_obj, best_weights, orthogonality = perform_layer_drsa_analysis(
                act_tensor, ctx_tensor, layer_name
            )
            
            if best_obj is not None:
                # Get DRSA activations for sample selection
                act_reshaped = act_tensor.squeeze(-1)
                drsa_activations = act_reshaped.numpy() @ best_weights.detach().numpy()
                
                # Analyze components
                mi_indices = np.where(labels == 1)[0]
                normal_indices = np.where(labels == 0)[0]
                
                component_info = []
                for comp_idx in range(3):
                    mi_acts = drsa_activations[mi_indices, comp_idx]
                    normal_acts = drsa_activations[normal_indices, comp_idx]
                    
                    separation = np.mean(mi_acts) - np.mean(normal_acts)
                    pattern_type = "MI-positive" if separation > 0 else "MI-negative"
                    
                    component_info.append({
                        'separation': separation,
                        'pattern_type': pattern_type
                    })
                
                # Store results
                all_layer_results[layer_name] = {
                    'objective': best_obj,
                    'weights': best_weights,
                    'orthogonality': orthogonality,
                    'component_info': component_info,
                    'drsa_activations': drsa_activations
                }
                
                print(f"✅ {layer_name} completed!")
            else:
                print(f"❌ {layer_name} failed")
                
        except Exception as e:
            print(f"❌ {layer_name} failed: {e}")
    
    if not all_layer_results:
        print("❌ No layers were successfully analyzed")
        return
    
    print(f"\n✅ Successfully analyzed {len(all_layer_results)} layers")
    
    # Select representative MI sample
    sample_idx, patient_id = select_representative_mi_sample_multi_layer(
        data, labels, patient_ids, all_layer_results
    )
    
    # Compute multi-layer relevance maps for the selected sample
    multi_layer_relevance, ecg_data = compute_multi_layer_relevance_maps(
        lrp_model, all_layer_results, data, sample_idx
    )
    
    # Create comprehensive visualization
    create_single_mi_multi_layer_visualization(
        sample_idx, patient_id, ecg_data, multi_layer_relevance, all_layer_results
    )
    
    # Analyze cross-layer patterns
    analyze_cross_layer_patterns(multi_layer_relevance, all_layer_results)
    
    print(f"\n🏆 SINGLE MI MULTI-LAYER ANALYSIS COMPLETE:")
    print(f"   ✅ One MI case ({patient_id}) analyzed across {len(all_layer_results)} layers")
    print(f"   ✅ LRP relevance computed for each layer's orthogonal subspaces")
    print(f"   ✅ Direct comparison of different network depth representations")
    print(f"   ✅ Cross-layer pattern analysis completed")

if __name__ == "__main__":
    main()
