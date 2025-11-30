#!/usr/bin/env python3
"""
Voice Clone Script with CosyVoice2
Supports both single text input and batch processing with SRT files

Usage: 
  Single mode: python voice_clone.py --text "Your text here" --reference /path/to/reference.mp3
  Batch mode:  python voice_clone.py --srt subtitles.srt --reference-dir /path/to/audio_dir
"""

import argparse
import os
import re
import sys

import torch
import torchaudio

# Add third_party path
sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed


def parse_srt(srt_file):
    """
    Parse SRT file and extract subtitle entries
    Returns list of dicts with 'id', 'start', 'end', 'text'
    """
    print(f"解析 SRT 文件: {srt_file}")

    with open(srt_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newlines to separate subtitle blocks
    blocks = re.split(r'\n\s*\n', content.strip())

    subtitles = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                # First line is the ID
                subtitle_id = int(lines[0].strip())

                # Second line is the timestamp
                timestamp = lines[1].strip()

                # Remaining lines are the text
                text = ' '.join(lines[2:]).strip()

                subtitles.append({
                    'id': subtitle_id,
                    'timestamp': timestamp,
                    'text': text
                })
            except (ValueError, IndexError) as e:
                print(f"警告: 跳过格式错误的块: {block[:50]}... 错误: {e}")
                continue

    print(f"解析了 {len(subtitles)} 条字幕")
    return subtitles


def find_reference_audio(reference_dir, subtitle_id, audio_prefix='segment'):
    """
    Find the reference audio file for a given subtitle ID
    Supports formats: segment_001.wav, segment_1.wav, etc.
    """
    # Try different naming patterns
    patterns = [
        f"{audio_prefix}_{subtitle_id:03d}.wav",  # segment_001.wav
        f"{audio_prefix}_{subtitle_id:03d}.mp3",
        f"{audio_prefix}_{subtitle_id}.wav",  # segment_1.wav
        f"{audio_prefix}_{subtitle_id}.mp3",
        f"{audio_prefix}{subtitle_id:03d}.wav",  # segment001.wav
        f"{audio_prefix}{subtitle_id:03d}.mp3",
        f"{audio_prefix}{subtitle_id}.wav",  # segment1.wav
        f"{audio_prefix}{subtitle_id}.mp3",
        f"{audio_prefix}_{subtitle_id:03d}.mp4",  # segment_001.mp4
        f"{audio_prefix}_{subtitle_id:03d}.m4a",
    ]

    for pattern in patterns:
        audio_path = os.path.join(reference_dir, pattern)
        if os.path.exists(audio_path):
            return audio_path

    return None


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='CosyVoice2 语音克隆 - 单条或批量处理模式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
单条模式示例:
  # 克隆单条中文文本
  python voice_clone.py --text "你好世界" --reference voice.mp3
  
  # 克隆英文文本
  python voice_clone.py --text "Hello world" --reference voice.mp3 --prompt-text "This is the reference"

批量模式示例:
  # 处理 SRT 文件
  python voice_clone.py --srt subtitles.srt --reference-dir ./audio_segments
  
  # 自定义音频文件前缀
  python voice_clone.py --srt subtitles.srt --reference-dir ./audio --audio-prefix audio
  
  # 使用自然语言控制（需要 Instruct 模型）
  python voice_clone.py --srt subtitles.srt --reference-dir ./audio --instruct "用温柔的声音说"

说明:
  - 支持零样本语音克隆（3s 极速复刻）
  - 支持跨语种复刻
  - 支持自然语言控制（需要 CosyVoice2-Instruct 模型）
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--text',
        type=str,
        help='[单条模式] 要合成的文本'
    )
    mode_group.add_argument(
        '--srt',
        type=str,
        help='[批量模式] SRT 字幕文件路径'
    )

    # Reference audio
    ref_group = parser.add_mutually_exclusive_group(required=True)
    ref_group.add_argument(
        '--reference',
        type=str,
        help='[单条模式] 参考音频文件路径（要克隆的声音）'
    )
    ref_group.add_argument(
        '--reference-dir',
        type=str,
        help='[批量模式] 包含参考音频文件的目录'
    )
    ref_group.add_argument(
        '--audio-path',
        type=str,
        help='[批量模式] 单个参考音频文件（所有字幕使用同一个）'
    )

    # Prompt text for reference audio
    parser.add_argument(
        '--prompt-text',
        type=str,
        default='',
        help='参考音频的文本内容（用于零样本克隆，可选）'
    )

    # Common arguments
    parser.add_argument(
        '--audio-prefix',
        type=str,
        default='segment',
        help='[批量模式] 音频文件前缀 (默认: segment)，文件命名如 segment_001.wav'
    )

    parser.add_argument(
        '--output-prefix',
        type=str,
        default='clone',
        help='[批量模式] 输出文件前缀 (默认: clone)，输出命名如 clone_001.wav'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='outputs_cosyvoice',
        help='输出目录'
    )

    parser.add_argument(
        '--model-dir',
        type=str,
        default='pretrained_models/CosyVoice2-0.5B',
        help='CosyVoice2 模型目录'
    )

    parser.add_argument(
        '--instruct',
        type=str,
        default='',
        help='自然语言控制指令（例如："用温柔的声音说"、"用四川话说"）'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='设备 (cuda:0, cpu 等)。如果不指定则自动检测'
    )

    parser.add_argument(
        '--load-jit',
        action='store_true',
        help='加载 JIT 模型以加速推理'
    )

    parser.add_argument(
        '--load-trt',
        action='store_true',
        help='加载 TensorRT 模型以加速推理（仅 GPU）'
    )

    parser.add_argument(
        '--load-vllm',
        action='store_true',
        help='使用 vLLM 加速 LLM 推理（仅 GPU）'
    )

    parser.add_argument(
        '--fp16',
        action='store_true',
        help='使用 FP16 精度（仅 GPU）'
    )

    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='[批量模式] 跳过已存在的输出文件'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（用于可重复性）'
    )

    return parser.parse_args()


def initialize_cosyvoice(model_dir, load_jit, load_trt, load_vllm, fp16):
    """Initialize CosyVoice2 model"""
    print(f"初始化 CosyVoice2 模型: {model_dir}")
    print(f"  JIT: {load_jit}, TRT: {load_trt}, vLLM: {load_vllm}, FP16: {fp16}")
    
    cosyvoice = CosyVoice2(
        model_dir,
        load_jit=load_jit,
        load_trt=load_trt,
        load_vllm=load_vllm,
        fp16=fp16
    )
    
    print(f"模型初始化成功，采样率: {cosyvoice.sample_rate} Hz")
    return cosyvoice


def synthesize_with_cosyvoice(cosyvoice, text, prompt_text, prompt_audio_path,
                               output_path, instruct_text='', seed=42):
    """
    Synthesize speech using CosyVoice2
    
    Args:
        cosyvoice: CosyVoice2 model instance
        text: Text to synthesize
        prompt_text: Text content of the reference audio
        prompt_audio_path: Path to reference audio file
        output_path: Output audio file path
        instruct_text: Natural language instruction (optional)
        seed: Random seed
    """
    # Load reference audio (16kHz)
    prompt_speech_16k = load_wav(prompt_audio_path, 16000)
    
    # Set random seed for reproducibility
    set_all_random_seed(seed)
    
    # Choose inference method based on instruct_text
    if instruct_text:
        # Use instruct mode (requires CosyVoice2-Instruct model)
        print(f"  使用指令模式: {instruct_text}")
        inference_gen = cosyvoice.inference_instruct2(
            text,
            instruct_text,
            prompt_speech_16k,
            stream=False
        )
    else:
        # Use zero-shot mode
        inference_gen = cosyvoice.inference_zero_shot(
            text,
            prompt_text,
            prompt_speech_16k,
            stream=False
        )
    
    # Get the synthesized audio
    for i, result in enumerate(inference_gen):
        tts_speech = result['tts_speech']
        torchaudio.save(output_path, tts_speech, cosyvoice.sample_rate)
        break  # Only take the first result


def process_single_mode(args, cosyvoice):
    """Process single text with single reference audio"""
    print("\n" + "=" * 60)
    print("单条模式")
    print("=" * 60)

    # Validate reference audio
    if not os.path.exists(args.reference):
        print(f"错误: 参考音频文件不存在: {args.reference}")
        return

    print(f"\n文本: {args.text}")
    print(f"参考音频: {args.reference}")
    if args.prompt_text:
        print(f"提示文本: {args.prompt_text}")
    if args.instruct:
        print(f"指令: {args.instruct}")

    try:
        output_filename = "output_single.wav"
        output_path = os.path.join(args.output, output_filename)

        synthesize_with_cosyvoice(
            cosyvoice=cosyvoice,
            text=args.text,
            prompt_text=args.prompt_text,
            prompt_audio_path=args.reference,
            output_path=output_path,
            instruct_text=args.instruct,
            seed=args.seed
        )

        print(f"  ✓ 已保存: {output_path}")

    except Exception as e:
        print(f"  ✗ 处理出错: {str(e)}")
        import traceback
        traceback.print_exc()


def process_batch_mode(args, cosyvoice):
    """Process SRT file with reference audio directory"""
    print("\n" + "=" * 60)
    print("批量模式")
    print("=" * 60)

    # Validate SRT file
    if not os.path.exists(args.srt):
        print(f"错误: SRT 文件不存在: {args.srt}")
        return

    # Validate reference directory or audio path
    if args.reference_dir and not os.path.isdir(args.reference_dir):
        if not (args.audio_path and os.path.exists(args.audio_path)):
            print(f"错误: 参考目录或音频路径不存在: {args.audio_path} {args.reference_dir}")
            return

    # Parse SRT file
    subtitles = parse_srt(args.srt)
    if not subtitles:
        print("错误: SRT 文件中没有有效字幕")
        return

    # Process statistics
    total = len(subtitles)
    success_count = 0
    skip_count = 0
    error_count = 0

    print(f"\n处理 {total} 条字幕...")
    print("-" * 60)

    # Determine if using single audio for all subtitles
    use_single_audio = args.audio_path is not None

    # Process each subtitle entry
    for idx, subtitle in enumerate(subtitles, 1):
        subtitle_id = subtitle['id']
        text = subtitle['text']

        print(f"\n[{idx}/{total}] ID: {subtitle_id}")
        print(f"文本: {text[:80]}{'...' if len(text) > 80 else ''}")

        # Find reference audio
        if use_single_audio:
            ref_audio = args.audio_path
        else:
            ref_audio = find_reference_audio(
                args.reference_dir,
                subtitle_id,
                args.audio_prefix
            )

        if not ref_audio:
            print(f"  ✗ 警告: 未找到 ID {subtitle_id} 的参考音频")
            error_count += 1
            continue

        try:
            # Generate output filename
            output_filename = f"{args.output_prefix}_{subtitle_id:03d}.wav"
            output_path = os.path.join(args.output, output_filename)

            # Skip if exists and flag is set
            if args.skip_existing and os.path.exists(output_path):
                print(f"  ⊘ 跳过: {output_filename} (已存在)")
                skip_count += 1
                continue

            synthesize_with_cosyvoice(
                cosyvoice=cosyvoice,
                text=text,
                prompt_text=args.prompt_text,
                prompt_audio_path=ref_audio,
                output_path=output_path,
                instruct_text=args.instruct,
                seed=args.seed + subtitle_id  # Different seed for each subtitle
            )

            print(f"  ✓ 已保存: {output_filename}")
            success_count += 1

        except Exception as e:
            print(f"  ✗ 处理 ID {subtitle_id} 时出错: {str(e)}")
            error_count += 1
            continue

    # Print summary
    print("\n" + "=" * 60)
    print("处理摘要")
    print("=" * 60)
    print(f"总条目:        {total}")
    print(f"成功处理:      {success_count}")
    print(f"跳过(已存在):  {skip_count}")
    print(f"错误:          {error_count}")
    if total > 0:
        print(f"成功率:        {success_count / total * 100:.1f}%")


def main():
    """Main execution function"""
    args = parse_args()

    # Determine device
    if args.device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 60)
    print("CosyVoice2 语音克隆脚本")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"输出目录: {args.output}")
    print(f"模型目录: {args.model_dir}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Initialize CosyVoice2
    cosyvoice = initialize_cosyvoice(
        args.model_dir,
        args.load_jit,
        args.load_trt,
        args.load_vllm,
        args.fp16
    )

    # Process based on mode
    if args.text:
        process_single_mode(args, cosyvoice)
    else:
        process_batch_mode(args, cosyvoice)

    print("\n" + "=" * 60)
    print("✓ 处理完成")
    print("=" * 60)
    print(f"输出文件已保存到: {args.output}")


if __name__ == "__main__":
    main()
