#!/bin/bash

# Checkpoint复制到U盘脚本
# 使用方法: ./copy_checkpoint_to_usb.sh

USB_DEVICE="/dev/sdc1"
MOUNT_POINT="/mnt/usb"
CHECKPOINT_DIR="/root/dp/StableSR_Canny/logs"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 显示菜单
show_menu() {
    clear
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}   Checkpoint 复制到U盘工具${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    # 显示U盘挂载状态
    echo -e "${CYAN}USB 状态信息：${NC}"
    if check_usb_mounted; then
        echo -e "  状态: ${GREEN}✓ 已挂载${NC}"
        echo -e "  挂载点: ${YELLOW}$MOUNT_POINT${NC}"
        
        # 显示磁盘空间信息
        local disk_info=$(df -h "$MOUNT_POINT" 2>/dev/null | tail -n 1)
        if [ -n "$disk_info" ]; then
            local total=$(echo "$disk_info" | awk '{print $2}')
            local used=$(echo "$disk_info" | awk '{print $3}')
            local avail=$(echo "$disk_info" | awk '{print $4}')
            local use_pct=$(echo "$disk_info" | awk '{print $5}')
            
            echo -e "  总容量: ${BLUE}$total${NC}"
            echo -e "  已使用: ${YELLOW}$used${NC} ($use_pct)"
            echo -e "  可用空间: ${GREEN}$avail${NC}"
        fi
    else
        echo -e "  状态: ${RED}✗ 未挂载${NC}"
        echo -e "  设备: ${YELLOW}$USB_DEVICE${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${CYAN}操作菜单：${NC}"
    echo -e "  ${GREEN}1.${NC} 复制 checkpoint"
    echo -e "  ${GREEN}2.${NC} 挂载 USB"
    echo -e "  ${GREEN}3.${NC} 卸载 USB"
    echo -e "  ${GREEN}4.${NC} 退出"
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -n -e "请选择操作 [1-4]: "
}

# 检查USB是否已挂载
check_usb_mounted() {
    if mount | grep -q "$MOUNT_POINT"; then
        return 0
    else
        return 1
    fi
}

# 挂载USB
mount_usb() {
    echo -e "${YELLOW}正在挂载USB...${NC}"
    
    # 检查是否已经挂载
    if check_usb_mounted; then
        echo -e "${GREEN}USB已经挂载在 $MOUNT_POINT${NC}"
        return 0
    fi
    
    # 检查挂载点目录是否存在
    if [ ! -d "$MOUNT_POINT" ]; then
        echo -e "${YELLOW}创建挂载点目录: $MOUNT_POINT${NC}"
        sudo mkdir -p "$MOUNT_POINT"
    fi
    
    # 检查设备是否存在
    if [ ! -e "$USB_DEVICE" ]; then
        echo -e "${RED}错误: 设备 $USB_DEVICE 不存在！${NC}"
        echo -e "${YELLOW}请检查U盘是否插入或设备路径是否正确${NC}"
        return 1
    fi
    
    # 挂载USB
    sudo mount "$USB_DEVICE" "$MOUNT_POINT"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}USB成功挂载到 $MOUNT_POINT${NC}"
        # 显示可用空间
        df -h "$MOUNT_POINT" | tail -n 1 | awk '{print "可用空间: " $4}'
        return 0
    else
        echo -e "${RED}挂载失败！${NC}"
        return 1
    fi
}

# 卸载USB
umount_usb() {
    echo -e "${YELLOW}正在卸载USB...${NC}"
    
    # 检查是否已挂载
    if ! check_usb_mounted; then
        echo -e "${YELLOW}USB未挂载${NC}"
        return 0
    fi
    
    # 卸载USB
    sudo umount "$MOUNT_POINT"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}USB已成功卸载${NC}"
        return 0
    else
        echo -e "${RED}卸载失败！可能有程序正在使用U盘${NC}"
        echo -e "${YELLOW}提示: 可以使用 'lsof | grep $MOUNT_POINT' 查看占用进程${NC}"
        return 1
    fi
}

# 解析用户选择的编号（支持单个、多个、范围）
parse_selection() {
    local input="$1"
    local max_num="$2"
    local selected_nums=()
    
    # 替换逗号为空格
    input="${input//,/ }"
    
    # 遍历每个token
    for token in $input; do
        # 检查是否是范围 (如 1-5)
        if [[ "$token" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            local start=${BASH_REMATCH[1]}
            local end=${BASH_REMATCH[2]}
            
            # 验证范围
            if [ $start -ge 1 ] && [ $end -le $max_num ] && [ $start -le $end ]; then
                for ((i=start; i<=end; i++)); do
                    selected_nums+=($i)
                done
            else
                echo "ERROR:范围 $token 无效"
                return 1
            fi
        # 检查是否是单个数字
        elif [[ "$token" =~ ^[0-9]+$ ]]; then
            if [ $token -ge 1 ] && [ $token -le $max_num ]; then
                selected_nums+=($token)
            else
                echo "ERROR:编号 $token 超出范围 (1-$max_num)"
                return 1
            fi
        else
            echo "ERROR:无效的输入格式: $token"
            return 1
        fi
    done
    
    # 去重并排序
    selected_nums=($(echo "${selected_nums[@]}" | tr ' ' '\n' | sort -nu | tr '\n' ' '))
    
    # 返回选中的编号
    echo "${selected_nums[@]}"
    return 0
}

# 列出可用的checkpoint
list_checkpoints() {
    echo -e "${YELLOW}正在扫描checkpoint目录...${NC}"
    echo ""
    
    # 查找所有checkpoint文件
    local checkpoints=()
    local checkpoint_times=()
    local checkpoint_sizes=()
    local checkpoint_dirs=()
    
    # 查找.ckpt和.pt文件
    while IFS= read -r -d '' file; do
        checkpoints+=("$file")
        checkpoint_times+=("$(stat -c %Y "$file")")
        checkpoint_sizes+=("$(du -h "$file" | cut -f1)")
        checkpoint_dirs+=("$(dirname "$file" | sed "s|$CHECKPOINT_DIR/||")")
    done < <(find "$CHECKPOINT_DIR" -type f \( -name "*.ckpt" -o -name "*.pt" -o -name "*.pth" \) -print0 2>/dev/null | sort -z)
    
    # 如果没有找到checkpoint
    if [ ${#checkpoints[@]} -eq 0 ]; then
        echo -e "${RED}未找到任何checkpoint文件！${NC}"
        return 1
    fi
    
    # 按目录组织并显示树形结构
    echo -e "${BLUE}$CHECKPOINT_DIR${NC}"
    local current_dir=""
    local idx=1
    
    for i in "${!checkpoints[@]}"; do
        local file="${checkpoints[$i]}"
        local dir="${checkpoint_dirs[$i]}"
        local size="${checkpoint_sizes[$i]}"
        local timestamp="${checkpoint_times[$i]}"
        local time_str=$(date -d "@$timestamp" "+%Y-%m-%d %H:%M:%S")
        local filename=$(basename "$file")
        
        # 获取类型信息
        local type_info=$(get_checkpoint_type "$file" 2>/dev/null)
        if [ $? -ne 0 ]; then
            type_info="${RED}未知${NC}"
        else
            type_info="${CYAN}[$type_info]${NC}"
        fi
        
        # 如果是新目录，显示目录名
        if [ "$dir" != "$current_dir" ]; then
            current_dir="$dir"
            echo -e "${BLUE}│${NC}"
            echo -e "${BLUE}├── $dir/${NC}"
        fi
        
        # 显示文件（树形结构）
        # 检查是否是该目录的最后一个文件
        local is_last=true
        for j in $(seq $((i+1)) $((${#checkpoints[@]}-1))); do
            if [ "${checkpoint_dirs[$j]}" = "$dir" ]; then
                is_last=false
                break
            fi
        done
        
        if [ "$is_last" = true ]; then
            echo -e "${BLUE}│   └──${NC} ${YELLOW}[$idx]${NC} $filename ${GREEN}[$size]${NC} $type_info ${BLUE}$time_str${NC}"
        else
            echo -e "${BLUE}│   ├──${NC} ${YELLOW}[$idx]${NC} $filename ${GREEN}[$size]${NC} $type_info ${BLUE}$time_str${NC}"
        fi
        
        ((idx++))
    done
    
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "总计: ${GREEN}${#checkpoints[@]}${NC} 个checkpoint文件"
    echo ""
    echo -e "0) 返回主菜单"
    echo ""
    echo -e "${BLUE}提示：${NC}"
    echo -e "  • 输入单个编号: ${GREEN}3${NC}"
    echo -e "  • 输入多个编号（逗号分隔）: ${GREEN}1,3,5${NC}"
    echo -e "  • 输入多个编号（空格分隔）: ${GREEN}1 3 5${NC}"
    echo -e "  • 输入范围: ${GREEN}1-5${NC}"
    echo -e "  • 输入 ${GREEN}a${NC} 复制所有"
    echo -e "  • 输入 ${GREEN}r${NC} 刷新列表"
    echo ""
    
    # 让用户选择
    while true; do
        echo -n "请选择要复制的checkpoint: "
        read choice
        
        if [ "$choice" = "0" ]; then
            return 0
        elif [ "$choice" = "r" ] || [ "$choice" = "R" ]; then
            # 刷新列表
            echo ""
            echo -e "${YELLOW}正在刷新checkpoint列表...${NC}"
            echo ""
            list_checkpoints
            return 0
        elif [ "$choice" = "a" ] || [ "$choice" = "A" ]; then
            # 复制所有checkpoint（类型自动从文件读取）
            echo ""
            echo -e "${YELLOW}准备批量复制所有 ${#checkpoints[@]} 个checkpoint${NC}"
            echo -e "${YELLOW}类型信息将自动从各checkpoint目录读取${NC}"
            echo ""
            
            echo -n -e "确认继续? [Y/n]: "
            read confirm
            if [ "$confirm" = "n" ] || [ "$confirm" = "N" ]; then
                echo -e "${YELLOW}已取消批量复制${NC}"
                return 0
            fi
            
            echo ""
            echo -e "${GREEN}开始批量复制...${NC}"
            echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            
            local success_count=0
            local fail_count=0
            
            for file in "${checkpoints[@]}"; do
                echo ""
                if copy_file_with_type "$file"; then
                    ((success_count++))
                else
                    ((fail_count++))
                fi
            done
            
            echo ""
            echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            echo -e "${GREEN}批量复制完成！${NC}"
            echo -e "  成功: ${GREEN}$success_count${NC} 个"
            if [ $fail_count -gt 0 ]; then
                echo -e "  失败: ${RED}$fail_count${NC} 个"
            fi
            echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            
            return 0
        else
            # 尝试解析多选或范围选择
            local parsed_nums=$(parse_selection "$choice" ${#checkpoints[@]})
            
            if [ $? -eq 0 ] && [ -n "$parsed_nums" ]; then
                # 将解析结果转换为数组
                local selected_array=($parsed_nums)
                local num_selected=${#selected_array[@]}
                
                echo ""
                echo -e "${GREEN}已选择 $num_selected 个checkpoint:${NC}"
                for num in "${selected_array[@]}"; do
                    local file="${checkpoints[$((num-1))]}"
                    local filename=$(basename "$file")
                    echo -e "  [$num] $filename"
                done
                echo ""
                
                # 确认是否继续
                echo -n -e "确认复制这些文件? [Y/n]: "
                read confirm
                if [ "$confirm" = "n" ] || [ "$confirm" = "N" ]; then
                    echo -e "${YELLOW}已取消${NC}"
                    continue
                fi
                
                # 批量复制（类型自动从文件读取）
                if [ $num_selected -gt 1 ]; then
                    echo ""
                    echo -e "${YELLOW}类型信息将自动从各checkpoint目录读取${NC}"
                    echo -e "${GREEN}开始批量复制...${NC}"
                    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
                    
                    local success_count=0
                    local fail_count=0
                    
                    for num in "${selected_array[@]}"; do
                        local file="${checkpoints[$((num-1))]}"
                        echo ""
                        if copy_file_with_type "$file"; then
                            ((success_count++))
                        else
                            ((fail_count++))
                        fi
                    done
                    
                    echo ""
                    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
                    echo -e "${GREEN}批量复制完成！${NC}"
                    echo -e "  成功: ${GREEN}$success_count${NC} 个"
                    if [ $fail_count -gt 0 ]; then
                        echo -e "  失败: ${RED}$fail_count${NC} 个"
                    fi
                    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
                    
                    return 0
                else
                    # 单个文件，使用详细复制
                    copy_file "${checkpoints[$((selected_array[0]-1))]}"
                    return 0
                fi
            else
                # 解析失败，显示错误
                if [[ "$parsed_nums" == ERROR:* ]]; then
                    echo -e "${RED}${parsed_nums#ERROR:}${NC}"
                else
                    echo -e "${RED}无效选择！${NC}"
                fi
            fi
        fi
    done
}

# 从checkpoint目录读取类型信息
get_checkpoint_type() {
    local source_file="$1"
    local checkpoint_dir=$(dirname "$source_file")
    
    # 在当前目录和上级目录查找configs目录
    local search_dir="$checkpoint_dir"
    local project_yaml=""
    
    # 最多向上查找3级目录
    for i in {1..3}; do
        # 查找configs目录中包含project的yaml文件
        if [ -d "$search_dir/configs" ]; then
            project_yaml=$(find "$search_dir/configs" -maxdepth 1 -type f -name "*project*.yaml" -o -name "*project*.yml" 2>/dev/null | head -n 1)
            
            if [ -n "$project_yaml" ]; then
                break
            fi
        fi
        
        # 移动到上级目录
        search_dir=$(dirname "$search_dir")
        
        # 如果已经到达logs目录或根目录，停止搜索
        if [ "$search_dir" = "$CHECKPOINT_DIR" ] || [ "$search_dir" = "/" ]; then
            break
        fi
    done
    
    if [ -z "$project_yaml" ] || [ ! -f "$project_yaml" ]; then
        echo "ERROR:未找到project配置文件"
        return 1
    fi
    
    # 从yaml文件中提取 lr_downscale_factor 和 train_with_edge
    local lr_downscale=$(grep -E "^\s*lr_downscale_factor:" "$project_yaml" | sed -E 's/.*lr_downscale_factor:\s*([0-9]+).*/\1/' | head -n 1)
    local train_edge=$(grep -E "^\s*train_with_edge:" "$project_yaml" | sed -E 's/.*train_with_edge:\s*(true|false).*/\1/I' | head -n 1)
    
    # 验证提取的参数
    if [ -z "$lr_downscale" ]; then
        echo "ERROR:无法读取lr_downscale_factor参数"
        return 1
    fi
    
    if [ -z "$train_edge" ]; then
        echo "ERROR:无法读取train_with_edge参数"
        return 1
    fi
    
    # 转换为首字母大写的格式 (True/False)
    if [[ "$train_edge" =~ ^[Tt]rue$ ]]; then
        train_edge="True"
    else
        train_edge="False"
    fi
    
    # 返回格式：4x_False, 8x_True 等
    echo "${lr_downscale}x_${train_edge}"
    return 0
}

# 复制文件到USB（自动读取类型）
copy_file_with_type() {
    local source_file="$1"
    
    # 检查USB是否挂载
    if ! check_usb_mounted; then
        echo -e "${RED}错误: USB未挂载！请先挂载USB${NC}"
        return 1
    fi
    
    # 检查源文件是否存在
    if [ ! -f "$source_file" ]; then
        echo -e "${RED}错误: 文件不存在: $source_file${NC}"
        return 1
    fi
    
    # 获取原文件信息
    local original_filename=$(basename "$source_file")
    local filesize=$(du -h "$source_file" | cut -f1)
    local file_ext="${original_filename##*.}"
    
    # 读取类型信息
    local type_info=$(get_checkpoint_type "$source_file")
    if [ $? -ne 0 ]; then
        echo -e "${RED}  ✗ ${type_info#ERROR:}，文件: $original_filename${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}复制: $original_filename ($filesize) -> ${type_info}${NC}"
    
    # 提取epoch信息（保留完整的数字）
    local epoch_num="000"
    if [[ "$original_filename" =~ epoch[=_-]?([0-9]+) ]]; then
        # 保留原始数字，去除前导零但至少保留3位
        local raw_num=${BASH_REMATCH[1]}
        epoch_num=$(printf "%03d" $((10#$raw_num)))
    elif [[ "$original_filename" =~ step[=_-]?([0-9]+) ]]; then
        epoch_num=$(printf "%03d" $((${BASH_REMATCH[1]} / 1000)))
    elif [[ "$original_filename" =~ ([0-9]{6,}) ]]; then
        epoch_num=$(printf "%03d" $((${BASH_REMATCH[1]} / 1000)))
    elif [[ "$original_filename" =~ last|final ]]; then
        epoch_num="final"
    fi
    
    # 获取文件修改时间
    local file_timestamp=$(stat -c %Y "$source_file")
    local time_str=$(date -d "@$file_timestamp" "+%Y%m%d_%H%M%S")
    
    # 生成新文件名：epoch_xxx_16x_False_时间.扩展名
    local new_filename="epoch_${epoch_num}_${type_info}_${time_str}.${file_ext}"
    
    # 创建ckpt目录
    local ckpt_dir="$MOUNT_POINT/ckpt"
    if [ ! -d "$ckpt_dir" ]; then
        mkdir -p "$ckpt_dir"
    fi
    
    local dest_file="$ckpt_dir/$new_filename"
    
    # 检查目标文件是否已存在（批量复制时自动跳过）
    if [ -f "$dest_file" ]; then
        echo -e "${YELLOW}  ⊙ 目标文件已存在，跳过${NC}"
        return 0
    fi
    
    # 检查U盘空间
    local available_space=$(df "$MOUNT_POINT" | tail -n 1 | awk '{print $4}')
    local file_size_kb=$(du -k "$source_file" | cut -f1)
    
    if [ $file_size_kb -gt $available_space ]; then
        echo -e "${RED}  ✗ 错误: U盘空间不足！${NC}"
        return 1
    fi
    
    # 复制文件
    rsync -ah --progress "$source_file" "$dest_file" 2>&1 | grep -v "^$"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        # 快速校验文件大小
        local src_size=$(stat -c %s "$source_file")
        local dst_size=$(stat -c %s "$dest_file")
        
        if [ "$src_size" = "$dst_size" ]; then
            echo -e "${GREEN}  ✓ 成功: $new_filename${NC}"
            return 0
        else
            echo -e "${RED}  ✗ 失败: 文件大小不匹配${NC}"
            rm -f "$dest_file"
            return 1
        fi
    else
        echo -e "${RED}  ✗ 复制失败${NC}"
        return 1
    fi
}

# 复制文件到USB（单个文件，带详细信息）
copy_file() {
    local source_file="$1"
    
    # 检查USB是否挂载
    if ! check_usb_mounted; then
        echo -e "${RED}错误: USB未挂载！请先挂载USB${NC}"
        return 1
    fi
    
    # 检查源文件是否存在
    if [ ! -f "$source_file" ]; then
        echo -e "${RED}错误: 文件不存在: $source_file${NC}"
        return 1
    fi
    
    # 获取原文件信息
    local original_filename=$(basename "$source_file")
    local filesize=$(du -h "$source_file" | cut -f1)
    local file_ext="${original_filename##*.}"
    
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}准备复制: $original_filename ($filesize)${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # 读取类型信息
    local type_info=$(get_checkpoint_type "$source_file")
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ ${type_info#ERROR:}${NC}"
        return 1
    fi
    
    echo -e "${BLUE}检测到类型: ${GREEN}$type_info${NC}"
    
    # 提取epoch信息（保留完整的数字）
    local epoch_num="000"
    if [[ "$original_filename" =~ epoch[=_-]?([0-9]+) ]]; then
        # 保留原始数字，去除前导零但至少保留3位
        local raw_num=${BASH_REMATCH[1]}
        epoch_num=$(printf "%03d" $((10#$raw_num)))
    elif [[ "$original_filename" =~ step[=_-]?([0-9]+) ]]; then
        epoch_num=$(printf "%03d" $((${BASH_REMATCH[1]} / 1000)))
    elif [[ "$original_filename" =~ ([0-9]{6,}) ]]; then
        epoch_num=$(printf "%03d" $((${BASH_REMATCH[1]} / 1000)))
    elif [[ "$original_filename" =~ last|final ]]; then
        epoch_num="final"
    fi
    
    # 获取文件修改时间
    local file_timestamp=$(stat -c %Y "$source_file")
    local time_str=$(date -d "@$file_timestamp" "+%Y%m%d_%H%M%S")
    
    # 生成新文件名：epoch_xxx_16x_False_时间.扩展名
    local new_filename="epoch_${epoch_num}_${type_info}_${time_str}.${file_ext}"
    
    # 创建ckpt目录
    local ckpt_dir="$MOUNT_POINT/ckpt"
    if [ ! -d "$ckpt_dir" ]; then
        echo -e "${YELLOW}创建目录: $ckpt_dir${NC}"
        mkdir -p "$ckpt_dir"
    fi
    
    local dest_file="$ckpt_dir/$new_filename"
    
    echo ""
    echo -e "${BLUE}复制信息：${NC}"
    echo -e "  源文件: $source_file"
    echo -e "  目标文件: $dest_file"
    echo -e "  类型: ${GREEN}$type_info${NC}"
    echo -e "  Epoch: ${GREEN}$epoch_num${NC}"
    echo -e "  大小: ${GREEN}$filesize${NC}"
    echo ""
    
    # 检查目标文件是否已存在
    if [ -f "$dest_file" ]; then
        echo -e "${YELLOW}⊙ 目标文件已存在，跳过复制${NC}"
        echo -e "  $new_filename"
        return 0
    fi
    
    # 检查U盘空间
    local available_space=$(df "$MOUNT_POINT" | tail -n 1 | awk '{print $4}')
    local file_size_kb=$(du -k "$source_file" | cut -f1)
    
    if [ $file_size_kb -gt $available_space ]; then
        echo -e "${RED}错误: U盘空间不足！${NC}"
        df -h "$MOUNT_POINT"
        return 1
    fi
    
    # 复制文件（带进度条）
    echo -e "${YELLOW}开始复制...${NC}"
    rsync -ah --progress "$source_file" "$dest_file"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}✓ 复制成功！${NC}"
        
        # 快速验证文件大小
        local src_size=$(stat -c %s "$source_file")
        local dst_size=$(stat -c %s "$dest_file")
        
        if [ "$src_size" = "$dst_size" ]; then
            echo -e "${GREEN}✓ 文件大小校验通过${NC}"
            echo -e "${GREEN}✓ 保存为: $new_filename${NC}"
        else
            echo -e "${RED}✗ 警告: 文件大小不匹配！${NC}"
        fi
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        
        return 0
    else
        echo -e "${RED}✗ 复制失败！${NC}"
        return 1
    fi
}

# 复制checkpoint主函数
copy_checkpoint() {
    clear
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}   复制 Checkpoint${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    list_checkpoints
}

# 主循环
main() {
    while true; do
        show_menu
        read choice
        
        case $choice in
            1)
                copy_checkpoint
                ;;
            2)
                mount_usb
                ;;
            3)
                umount_usb
                ;;
            4)
                echo -e "${GREEN}退出程序${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}无效选择！请输入 1-4${NC}"
                ;;
        esac
        
        echo ""
        echo -n "按任意键继续..."
        read -n 1
    done
}

# 启动主程序
main

